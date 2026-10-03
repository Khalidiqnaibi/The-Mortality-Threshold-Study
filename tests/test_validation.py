"""Validation tests V1-V8 (protocol Section 12) + leakage test. Small H for speed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy import stats

import src  # noqa: F401  (PRNG config)
from src.config import HP, Sig, Dyn
from src.device import sample_device, sample_many, N_IN, N_OUT
from src.substrate import relax, energy, forces, rho
from src.eqprop import (init_params, effective, three_phases, eqprop_update)
from src.train import train_runs, evaluate, accuracy, SOFT, LABEL

H = 32
SIG, DYN = Sig(), Dyn()
CLEAN = Dyn(sigma_r=0, sigma_prog=0, sigma_w=0, alpha=0)


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, size=(n, N_IN), dtype=np.uint8)
    W = rng.normal(size=(N_IN, 10))
    Y = (X.astype(np.float32) @ W).argmax(1).astype(np.int32)
    return X, Y


def _deterministic_hp(beta, steps=600):
    return HP(H=H, T=0.0, dt=0.2, beta=beta, n_burn_free=steps, n_samp_free=1,
              n_burn_nudge=steps, n_samp_nudge=1)


def _exact_and_ep(beta, key=jax.random.PRNGKey(3)):
    hp = _deterministic_hp(beta)
    dev = sample_device(key, 0.0, SIG, CLEAN, H)
    params = init_params(jax.random.PRNGKey(4), H, hp)
    params = jax.tree_util.tree_map(lambda a: a * 3.0, params)  # non-trivial net
    params["bh"] = 0.1 * jax.random.normal(jax.random.PRNGKey(5), (H,))
    X, Y = _data(16)
    xin = jnp.asarray(X, jnp.float32) / 255.0
    y = jax.nn.one_hot(Y, 10)

    def loss(p):
        W1e, W2e, bhe, boe = effective(p, dev, hp)
        rin = rho(xin, dev["a_in"], dev["k_in"], dev["c_in"])
        (xh, xo), _ = relax(jnp.zeros((16, H)), jnp.zeros((16, N_OUT)), rin @ W1e,
                            W2e, bhe, boe, dev, 0.0, y, key, 0.0, hp.dt, 600, 1)
        return 0.5 * ((xo - y) ** 2).sum(-1).mean()

    g = jax.grad(loss)(params)
    rin, sf, sp, sm = three_phases(params, dev, xin, y, key, hp, CLEAN)
    ep = eqprop_update(rin, sp, sm, beta)
    flat = lambda d: jnp.concatenate([d[k].ravel() for k in ("W1", "W2", "bh", "bo")])
    return -flat(g), flat(ep)


def test_V3_eqprop_matches_gradient():
    true, est = _exact_and_ep(0.05)
    cos = float(true @ est / (jnp.linalg.norm(true) * jnp.linalg.norm(est)))
    assert cos > 0.99, cos


def test_V8_symmetric_bias_scales_beta_squared():
    betas = [0.4, 0.2, 0.1]
    errs = []
    for b in betas:
        true, est = _exact_and_ep(b)
        errs.append(float(jnp.linalg.norm(est - true) / jnp.linalg.norm(true)))
    slope = np.polyfit(np.log(betas), np.log(errs), 1)[0]
    assert slope > 1.6, (slope, errs)


def test_V4_noise_free_relaxation_lowers_energy():
    hp = HP(H=H)
    dev = sample_device(jax.random.PRNGKey(1), 1.0, SIG, DYN, H)
    p = init_params(jax.random.PRNGKey(2), H, hp)
    p = jax.tree_util.tree_map(lambda a: 3 * a, p)
    W1e, W2e, bhe, boe = effective(p, dev, hp)
    X, Y = _data(8)
    xin = jnp.asarray(X, jnp.float32) / 255.0
    rin = rho(xin, dev["a_in"], dev["k_in"], dev["c_in"])
    y = jax.nn.one_hot(Y, 10)
    xh, xo = jnp.zeros((8, H)), jnp.zeros((8, 10))
    Es = []
    for _ in range(200):
        Es.append(np.asarray(energy(xh, xo, rin, W1e, W2e, bhe, boe, dev, 0.3, y)))
        gh, go = forces(xh, xo, rin @ W1e, W2e, bhe, boe, dev, 0.3, y)
        xh, xo = xh - 0.05 * gh, xo - 0.05 * go
    Es = np.stack(Es)
    assert (np.diff(Es, axis=0) <= 1e-5).all()


@pytest.mark.parametrize("dt", [0.05, 0.25, 0.5])
def test_V5_harmonic_stationary_variance(dt):
    # zero couplings & biases => force = x (E = 1/2 x^2): OU process.
    T, B = 0.01, 4000
    dev = sample_device(jax.random.PRNGKey(0), 0.0, SIG, DYN, 64)
    z = lambda *s: jnp.zeros(s)
    (xh, _), _ = relax(z(B, 64), z(B, 10), z(B, 64), z(64, 10), z(64), z(10), dev,
                       0.0, z(B, 10), jax.random.PRNGKey(1), T, dt, int(20 / dt), 1)
    v = float(jnp.var(xh))
    theory = 2 * T / (2 - dt)  # exact Euler-Maruyama stationary variance
    assert abs(v - theory) / theory < 0.02, (v, theory)
    if dt >= 0.25:
        assert v > T * 1.05  # step-size bias visible


def test_V6_device_sampler():
    s = 2.0
    d = sample_device(jax.random.PRNGKey(7), s, SIG, DYN, 256)
    z = np.log(np.asarray(d["g1"])).ravel() / (s * SIG.g)
    assert stats.kstest(z, "norm").pvalue > 0.001
    zc = np.asarray(d["c_h"]) / (s * SIG.c)
    assert stats.kstest(zc, "norm").pvalue > 0.001
    d0 = sample_device(jax.random.PRNGKey(7), 0.0, SIG, DYN, 256)
    for k in ("g1", "g2", "gbh", "gbo", "a_in", "k_in", "a_h", "k_h", "a_o", "k_o",
              "T_h", "T_o"):
        assert np.all(np.asarray(d0[k]) == 1.0), k
    for k in ("o1", "o2", "c_in", "c_h", "c_o"):
        assert np.all(np.asarray(d0[k]) == 0.0), k


def _small_train(seed=0, updates=100):
    hp = HP(H=H, T=0.002, lr1=0.2, lr2=0.1)
    X, Y = _data(2000)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    devs = sample_many(keys, 0.0, SIG, DYN, H)
    p0 = jax.vmap(lambda k: init_params(k, H, hp))(keys)
    p, _ = train_runs(p0, keys, jnp.asarray(X), jnp.asarray(Y), hp, SIG, DYN,
                      updates, devs=devs)
    return hp, p, devs, X, Y


def test_V1_V2_zero_mismatch_and_identical_device_port():
    hp, p, devs, X, Y = _small_train()
    # V1: s=0 devices with different seeds are identical => port is exact
    k = jax.random.split(jax.random.PRNGKey(9), 2)
    own = evaluate(p, devs, k, X[:1000], hp, DYN)
    swapped = jax.tree_util.tree_map(lambda a: a[::-1], devs)
    port = evaluate(p, swapped, k, X[:1000], hp, DYN)
    np.testing.assert_array_equal(own, port)
    # V2: same device seed at s>0 => bit-identical device; same seed + same
    # config => bit-identical outputs (evaluated as separate runs)
    from src.device import devices_from_seeds
    d = devices_from_seeds([123, 123, 7], 1.0, SIG, DYN, H)
    for kk in d:
        np.testing.assert_array_equal(np.asarray(d[kk][0]), np.asarray(d[kk][1]))
    # a device is a pure function of its seed, independent of batch size
    d_alone = devices_from_seeds([7], 1.0, SIG, DYN, H)
    np.testing.assert_array_equal(np.asarray(d["g1"][2]), np.asarray(d_alone["g1"][0]))
    one = lambda t: jax.tree_util.tree_map(lambda a: a[:1], t)
    o1 = evaluate(one(p), one(d), k[:1], X[:1000], hp, DYN)
    o2 = evaluate(one(p), one(d), k[:1], X[:1000], hp, DYN)
    np.testing.assert_array_equal(o1, o2)


def test_V7_reproducibility():
    _, p1, _, _, _ = _small_train(seed=3, updates=50)
    _, p2, _, _, _ = _small_train(seed=3, updates=50)
    for k in p1:
        np.testing.assert_array_equal(np.asarray(p1[k]), np.asarray(p2[k]))


def test_leakage_student_depends_only_on_readouts():
    """Output-only distillation: identical readouts + identical student init =>
    identical student, whatever the teacher's weights were. The trainer has no
    argument through which teacher parameters could enter."""
    hp, p, devs, X, Y = _small_train(updates=50)
    readouts = jnp.asarray(np.random.default_rng(0).random((1, len(X), 10)), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(5), 2)
    sdev = sample_many(keys, 1.0, SIG, DYN, H)
    init = jax.vmap(lambda k: init_params(k, H, hp))(keys)
    run = lambda: train_runs(init, keys, jnp.asarray(X), jnp.asarray(Y), hp, SIG, DYN,
                             50, devs=sdev, mode=[SOFT, SOFT], readouts=readouts)[0]
    a, b = run(), run()
    import inspect
    assert "teacher" not in " ".join(inspect.signature(train_runs).parameters)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
        assert not np.allclose(np.asarray(a[k]), np.asarray(p[k]))
