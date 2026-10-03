"""Device degradation model v1 (protocol Section 5, with review fixes).

A device is a fixed pytree of random draws made once from a device key.
Only *static* spreads are multiplied by s (fix: dynamic noise held fixed so the
mortality curve isolates device-specific mismatch). The programmed parameters
(W1, W2, bh, bo) are what gets copied when porting; everything here stays with
the device.

E2  w_eff = g * w_prog(t) + o * w_max,   g = exp(s*sig_g*eps)
E3  rho_i(x) = a_i * tanh(k_i * x + c_i)   (k_i = steepness; renamed from b_i to
                                             avoid the clash with biases)
E4  T_i = T * exp(s*sig_T*zeta_i)
E8  w_prog(t) = w_prog(t0) * (t/t0)^(-nu)  (drift acts on the programmed
                                             conductance, before the gain: E9)
"""
import jax
import jax.numpy as jnp

N_IN, N_OUT = 784, 10


def sample_device(key, s, sig, dyn, H):
    ks = jax.random.split(key, 20)
    n = lambda k, shape: jax.random.normal(k, shape, jnp.float32)
    ln = lambda k, shape, sd: jnp.exp(s * sd * n(k, shape))
    return dict(
        g1=ln(ks[0], (N_IN, H), sig.g), o1=s * sig.o * n(ks[1], (N_IN, H)),
        g2=ln(ks[2], (H, N_OUT), sig.g), o2=s * sig.o * n(ks[3], (H, N_OUT)),
        gbh=ln(ks[4], (H,), sig.g), gbo=ln(ks[5], (N_OUT,), sig.g),
        a_in=ln(ks[6], (N_IN,), sig.a), k_in=ln(ks[7], (N_IN,), sig.b),
        c_in=s * sig.c * n(ks[8], (N_IN,)),
        a_h=ln(ks[9], (H,), sig.a), k_h=ln(ks[10], (H,), sig.b),
        c_h=s * sig.c * n(ks[11], (H,)),
        a_o=ln(ks[12], (N_OUT,), sig.a), k_o=ln(ks[13], (N_OUT,), sig.b),
        c_o=s * sig.c * n(ks[14], (N_OUT,)),
        T_h=ln(ks[15], (H,), sig.T), T_o=ln(ks[16], (N_OUT,), sig.T),
        nu1=jnp.clip(dyn.nu_bar + dyn.sigma_nu * n(ks[17], (N_IN, H)), 0.0),
        nu2=jnp.clip(dyn.nu_bar + dyn.sigma_nu * n(ks[18], (H, N_OUT)), 0.0),
    )


def identity_device(H, dyn):
    return sample_device(jax.random.PRNGKey(0), 0.0, _zero_sig(), dyn, H)


def _zero_sig():
    from .config import Sig
    return Sig(0, 0, 0, 0, 0, 0)


def stack(devs):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *devs)


def sample_many(keys, s, sig, dyn, H):
    return jax.vmap(lambda k: sample_device(k, s, sig, dyn, H))(keys)


def threefry_keys(seeds):
    """Typed threefry keys: vmap-consistent, so a device is a pure function of
    its integer seed regardless of how many devices are sampled together."""
    return jax.vmap(lambda s: jax.random.key(s, impl="threefry2x32"))(
        jnp.asarray(seeds, jnp.uint32))


def devices_from_seeds(seeds, s, sig, dyn, H):
    return jax.vmap(lambda k: sample_device(k, s, sig, dyn, H))(threefry_keys(seeds))
