"""In-situ symmetric EqProp with device-level read/program non-idealities.

E13  dw_ij = eta/(2 beta) * (<rho_i rho_j>_{+beta} - <rho_i rho_j>_{-beta})

The default rule uses the products the device actually produces (mismatched
rho) and omits the coupling gain g_ij (which the learning circuit cannot see).
With g_ij > 0 this is a positive diagonal preconditioning of the true gradient
w.r.t. the programmed weight. hp.oracle=True multiplies by g_ij (ablation 13a).

Read noise: one multiplicative draw per relaxation run (per minibatch), shared
by the free, +beta and -beta phases of that update. Thermal noise is
independent across phases (physical).
"""
import jax
import jax.numpy as jnp
from .substrate import relax, rho, Stats
from .device import N_IN, N_OUT


def init_params(key, H, hp):
    k1, k2 = jax.random.split(key)
    lim1 = 1.0 / jnp.sqrt(N_IN)
    lim2 = 1.0 / jnp.sqrt(H)
    return dict(
        W1=jax.random.uniform(k1, (N_IN, H), minval=-lim1, maxval=lim1),
        W2=jax.random.uniform(k2, (H, N_OUT), minval=-lim2, maxval=lim2),
        bh=jnp.zeros((H,)), bo=jnp.zeros((N_OUT,)))


def effective(params, dev, hp, t_ratio=1.0):
    """E9: W_dyn = Drift(W_prog) * g + o. t_ratio = t/t0 (1 = no drift)."""
    W1 = params["W1"] * t_ratio ** (-dev["nu1"])
    W2 = params["W2"] * t_ratio ** (-dev["nu2"])
    W1e = dev["g1"] * W1 + dev["o1"] * hp.wmax1
    W2e = dev["g2"] * W2 + dev["o2"] * hp.wmax2
    return W1e, W2e, dev["gbh"] * params["bh"], dev["gbo"] * params["bo"]


def _read(W, key, sr):
    return W * (1.0 + sr * jax.random.normal(key, W.shape))


def free_phase(params, dev, xin, key, hp, dyn, dyn_mult=1.0, t_ratio=1.0,
               n_burn=None, n_samp=None):
    """Inference: stationary mean of the output nodes. xin in [0,1]."""
    W1e, W2e, bhe, boe = effective(params, dev, hp, t_ratio)
    k1, k2, kf = jax.random.split(key, 3)
    sr = dyn.sigma_r * dyn_mult
    W1r, W2r = _read(W1e, k1, sr), _read(W2e, k2, sr)
    rin = rho(xin, dev["a_in"], dev["k_in"], dev["c_in"])
    B, H = xin.shape[0], W1e.shape[1]
    y0 = jnp.zeros((B, N_OUT))
    _, st = relax(jnp.zeros((B, H)), jnp.zeros((B, N_OUT)), rin @ W1r, W2r,
                  bhe, boe, dev, 0.0, y0, kf, hp.T, hp.dt,
                  hp.n_eval_burn if n_burn is None else n_burn,
                  hp.n_eval_samp if n_samp is None else n_samp)
    return st.xo


def three_phases(params, dev, xin, y, key, hp, dyn, dyn_mult=1.0):
    W1e, W2e, bhe, boe = effective(params, dev, hp)
    k1, k2, kf, kp, km = jax.random.split(key, 5)
    sr = dyn.sigma_r * dyn_mult
    W1r, W2r = _read(W1e, k1, sr), _read(W2e, k2, sr)
    rin = rho(xin, dev["a_in"], dev["k_in"], dev["c_in"])
    drive = rin @ W1r
    B, H = xin.shape[0], W1e.shape[1]
    (xh, xo), sf = relax(jnp.zeros((B, H)), jnp.zeros((B, N_OUT)), drive, W2r,
                         bhe, boe, dev, 0.0, y, kf, hp.T, hp.dt,
                         hp.n_burn_free, hp.n_samp_free)
    _, sp = relax(xh, xo, drive, W2r, bhe, boe, dev, hp.beta, y, kp, hp.T,
                  hp.dt, hp.n_burn_nudge, hp.n_samp_nudge)
    _, sm = relax(xh, xo, drive, W2r, bhe, boe, dev, -hp.beta, y, km, hp.T,
                  hp.dt, hp.n_burn_nudge, hp.n_samp_nudge)
    return rin, sf, sp, sm


def eqprop_update(rin, sp: Stats, sm: Stats, beta):
    """Ascent direction (= minus the loss gradient estimate)."""
    B = rin.shape[0]
    dh = sp.rh - sm.rh
    return dict(W1=rin.T @ dh / (2 * beta * B),
                W2=(sp.C - sm.C) / (2 * beta * B),
                bh=dh.mean(0) / (2 * beta),
                bo=(sp.ro - sm.ro).mean(0) / (2 * beta))


def program(W, dW, key, wmax, dyn, dyn_mult):
    """E6/E6a soft-bound asymmetric update + programming/write noise; E7 opt."""
    up = dW > 0
    F = jnp.where(up, (1 + dyn.alpha) * (1 - W / wmax),
                  (1 - dyn.alpha) * (1 + W / wmax))
    k1, k3 = jax.random.split(key)
    # sigma_prog*|dW|*xi + sigma_w*wmax*xi' with independent xi, xi' is exactly
    # N(0, sigma_prog^2 dW^2 + sigma_w^2 wmax^2): one draw instead of two.
    sd = dyn_mult * jnp.sqrt((dyn.sigma_prog * dW) ** 2 + (dyn.sigma_w * wmax) ** 2)
    dWa = dW * F + sd * jax.random.normal(k1, W.shape)
    Wn = jnp.clip(W + dWa, -wmax, wmax)
    if dyn.levels:
        dq = 2 * wmax / (dyn.levels - 1)
        u = jax.random.uniform(k3, W.shape)
        Wn = jnp.clip(dq * jnp.floor(Wn / dq + u), -wmax, wmax)  # stochastic rounding
    return Wn


def train_step(params, dev, xin, y, key, lr_scale, hp, dyn, dyn_mult=1.0):
    kph, kw1, kw2, kb1, kb2 = jax.random.split(key, 5)
    rin, sf, sp, sm = three_phases(params, dev, xin, y, kph, hp, dyn, dyn_mult)
    g = eqprop_update(rin, sp, sm, hp.beta)
    if hp.oracle:
        g = dict(W1=g["W1"] * dev["g1"], W2=g["W2"] * dev["g2"],
                 bh=g["bh"] * dev["gbh"], bo=g["bo"] * dev["gbo"])
    lr1, lr2 = hp.lr1 * lr_scale, hp.lr2 * lr_scale
    new = dict(
        W1=program(params["W1"], lr1 * g["W1"], kw1, hp.wmax1, dyn, dyn_mult),
        W2=program(params["W2"], lr2 * g["W2"], kw2, hp.wmax2, dyn, dyn_mult),
        bh=program(params["bh"], lr1 * g["bh"], kb1, hp.wmaxb, dyn, dyn_mult),
        bo=program(params["bo"], lr2 * g["bo"], kb2, hp.wmaxb, dyn, dyn_mult))
    loss = 0.5 * ((sf.xo - y) ** 2).sum(-1).mean()
    return new, loss


def init_from_seeds(seeds, H, hp):
    from .device import threefry_keys
    return jax.vmap(lambda k: init_params(k, H, hp))(threefry_keys(seeds))
