"""Batched in-situ training of many independent runs at once (vmap over runs).

One generic trainer covers every condition in the study:
  * scratch / teacher training      mode=LABEL, n_avail = 60000
  * output-only distillation        mode=SOFT (teacher stationary-mean readout)
                                    mode=HARD (one-hot of teacher argmax)
  * labeled-n baseline              mode=LABEL, n_avail = n
  * robust-once-and-copy teacher    robust=True: a fresh device is drawn from
                                    the device distribution at every update
Every run of one call gets the same number of updates (compute matched): the
student sees n_avail transfer images, sampled with replacement, for U updates.

Leakage guarantee: the trainer receives only the teacher *readouts* (an array
of output means on the transfer images), never teacher parameters.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .eqprop import train_step, free_phase
from .device import sample_device

LABEL, SOFT, HARD = 0, 1, 2


def _targets(mode, Y, readouts, tidx, idx):
    lab = jax.nn.one_hot(Y[idx], 10)
    soft = readouts[tidx, idx]
    hard = jax.nn.one_hot(jnp.argmax(soft, -1), 10)
    return jnp.where(mode == LABEL, lab, jnp.where(mode == SOFT, soft, hard))


def make_chunk(hp, sig, dyn, chunk, total, robust):
    B, H = hp.batch, hp.H

    def one_run(params, key, dev, dkey, s, n_avail, mode, tidx, dm,
                step0, X, Y, perm, readouts):
        def body(carry, i):
            params, key = carry
            key, kb, kt = jax.random.split(key, 3)
            u = step0 + i
            pos = jnp.floor(jax.random.uniform(kb, (B,)) * n_avail).astype(jnp.int32)
            idx = perm[pos]
            xin = X[idx].astype(jnp.float32) / 255.0
            y = _targets(mode, Y, readouts, tidx, idx)
            d = sample_device(jax.random.fold_in(dkey, u), s, sig, dyn, H) if robust else dev
            lr_scale = 1.0 - 0.9 * u / total
            params, loss = train_step(params, d, xin, y, kt, lr_scale, hp, dyn, dm)
            return (params, key), loss

        (params, key), losses = lax.scan(body, (params, key), jnp.arange(chunk))
        return params, key, losses.mean()

    dev_axis = None if robust else 0
    v = jax.vmap(one_run, in_axes=(0, 0, dev_axis, 0, 0, 0, 0, 0, 0,
                                   None, None, None, None, None))
    return jax.jit(v)


def train_runs(params0, keys, X, Y, hp, sig, dyn, updates, *, devs=None,
               dev_keys=None, s=None, n_avail=None, mode=None, tidx=None,
               dyn_mult=None, perm=None, readouts=None, robust=False,
               chunk=50, log=None, tag=""):
    R = keys.shape[0]
    full = lambda v, dt: jnp.full((R,), v, dt)
    n_avail = full(X.shape[0], jnp.int32) if n_avail is None else jnp.asarray(n_avail, jnp.int32)
    mode = full(LABEL, jnp.int32) if mode is None else jnp.asarray(mode, jnp.int32)
    tidx = full(0, jnp.int32) if tidx is None else jnp.asarray(tidx, jnp.int32)
    dyn_mult = full(1.0, jnp.float32) if dyn_mult is None else jnp.asarray(dyn_mult, jnp.float32)
    s = full(0.0, jnp.float32) if s is None else jnp.asarray(s, jnp.float32)
    dev_keys = keys if dev_keys is None else dev_keys
    perm = jnp.arange(X.shape[0]) if perm is None else perm
    readouts = jnp.zeros((1, X.shape[0], 10)) if readouts is None else readouts
    assert updates % chunk == 0
    fn = make_chunk(hp, sig, dyn, chunk, updates, robust)
    params, t0, losses = params0, time.time(), []
    for c in range(updates // chunk):
        params, keys, l = fn(params, keys, devs, dev_keys, s, n_avail, mode,
                             tidx, dyn_mult, c * chunk, X, Y, perm, readouts)
        losses.append(np.asarray(l))
        if log and (c == 0 or (c + 1) % max(1, (updates // chunk) // 5) == 0):
            log(f"{tag} upd {(c+1)*chunk}/{updates} loss {np.nanmean(losses[-1]):.4f} "
                f"t={time.time()-t0:.0f}s")
    return params, np.stack(losses, 1)  # (R, n_chunks)


def make_eval(hp, dyn):
    def one(params, dev, key, dm, tr, Xb):
        def body(k, xb):
            k, kk = jax.random.split(k)
            return k, free_phase(params, dev, xb.astype(jnp.float32) / 255.0,
                                 kk, hp, dyn, dm, tr)
        _, xo = lax.scan(body, key, Xb)
        return xo.reshape(-1, 10)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0, None)))


def evaluate(params, devs, keys, X, hp, dyn, *, dyn_mult=None, t_ratio=None,
             batch=500, max_runs=None):
    """Returns stationary-mean outputs (R, N, 10). X: uint8 (N,784)."""
    R = keys.shape[0]
    N = X.shape[0]
    assert N % batch == 0
    Xb = jnp.asarray(X).reshape(N // batch, batch, -1)
    dm = jnp.ones((R,)) if dyn_mult is None else jnp.asarray(dyn_mult, jnp.float32)
    tr = jnp.ones((R,)) if t_ratio is None else jnp.asarray(t_ratio, jnp.float32)
    fn = make_eval(hp, dyn)
    step = max_runs or R
    outs = []
    for i in range(0, R, step):
        sl = lambda t: jax.tree_util.tree_map(lambda a: a[i:i + step], t)
        outs.append(np.asarray(fn(sl(params), sl(devs), keys[i:i + step],
                                  dm[i:i + step], tr[i:i + step], Xb)))
    return np.concatenate(outs, 0)


def accuracy(out, y):
    return (out.argmax(-1) == y[None]).mean(-1)
