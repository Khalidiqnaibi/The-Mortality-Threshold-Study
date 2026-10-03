"""Layered nonlinear Hopfield-style energy and Euler-Maruyama Langevin relaxation.

E10  E = 1/2 sum x^2 - sum_{edges} w_ij rho_i rho_j - sum b_i rho_i
E11  E_beta = E + beta * 1/2 ||x_out - y||^2
E12  dx = -dE_beta/dx dt + sqrt(2 T_i dt) eta

The input layer is clamped to the image; hidden and output nodes are free.
With per-node temperatures T_i the stationary law is not exactly Boltzmann in E
(non-equilibrium steady state); this is part of the device model and the reason
the EqProp estimator is only exact at s = 0 (see V3 and the paper discussion).
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax


def rho(x, a, k, c):
    return a * jnp.tanh(k * x + c)


def rho_d(x, a, k, c):
    t = jnp.tanh(k * x + c)
    return a * t, a * k * (1.0 - t * t)


class Stats(NamedTuple):
    rh: jnp.ndarray    # (B,H) time-mean rho_h
    ro: jnp.ndarray    # (B,O) time-mean rho_o
    xo: jnp.ndarray    # (B,O) time-mean x_o   (the readout)
    C: jnp.ndarray     # (H,O) batch-summed time-mean rho_h rho_o^T


def energy(xh, xo, rin, W1e, W2e, bhe, boe, dev, beta, y):
    rh = rho(xh, dev["a_h"], dev["k_h"], dev["c_h"])
    ro = rho(xo, dev["a_o"], dev["k_o"], dev["c_o"])
    E = (0.5 * (xh ** 2).sum(-1) + 0.5 * (xo ** 2).sum(-1)
         - ((rin @ W1e) * rh).sum(-1) - ((rh @ W2e) * ro).sum(-1)
         - (rh * bhe).sum(-1) - (ro * boe).sum(-1))
    return E + beta * 0.5 * ((xo - y) ** 2).sum(-1)


def forces(xh, xo, drive_h, W2e, bhe, boe, dev, beta, y):
    rh, dh = rho_d(xh, dev["a_h"], dev["k_h"], dev["c_h"])
    ro, do = rho_d(xo, dev["a_o"], dev["k_o"], dev["c_o"])
    gh = xh - dh * (drive_h + ro @ W2e.T + bhe)
    go = xo - do * (rh @ W2e + boe) + beta * (xo - y)
    return gh, go


def relax(xh, xo, drive_h, W2e, bhe, boe, dev, beta, y, key, T, dt,
          n_burn, n_samp):
    """Burn-in n_burn steps, then average over n_samp steps (thinning 1)."""
    sh = jnp.sqrt(2.0 * T * dev["T_h"] * dt)
    so = jnp.sqrt(2.0 * T * dev["T_o"] * dt)

    def step(state, k):
        xh, xo = state
        gh, go = forces(xh, xo, drive_h, W2e, bhe, boe, dev, beta, y)
        k1, k2 = jax.random.split(k)
        xh = xh - dt * gh + sh * jax.random.normal(k1, xh.shape)
        xo = xo - dt * go + so * jax.random.normal(k2, xo.shape)
        return (xh, xo)

    kb, ks = jax.random.split(key)
    state = (xh, xo)
    if n_burn > 0:
        state, _ = lax.scan(lambda s, k: (step(s, k), None), state,
                            jax.random.split(kb, n_burn))
    H, O = W2e.shape
    B = xh.shape[0]
    acc0 = Stats(jnp.zeros((B, H)), jnp.zeros((B, O)), jnp.zeros((B, O)),
                 jnp.zeros((H, O)))

    def samp(carry, k):
        state, acc = carry
        state = step(state, k)
        xh, xo = state
        rh = rho(xh, dev["a_h"], dev["k_h"], dev["c_h"])
        ro = rho(xo, dev["a_o"], dev["k_o"], dev["c_o"])
        acc = Stats(acc.rh + rh, acc.ro + ro, acc.xo + xo, acc.C + rh.T @ ro)
        return (state, acc), None

    (state, acc), _ = lax.scan(samp, (state, acc0), jax.random.split(ks, n_samp))
    acc = jax.tree_util.tree_map(lambda a: a / n_samp, acc)
    return state, acc
