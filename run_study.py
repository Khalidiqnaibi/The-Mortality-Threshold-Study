"""Study 6 driver. Resumable stages; every stage writes to results/<cfg>/.

  python run_study.py <config.yaml> pilot      # s in {0,1,4}, 3x3, pilot seeds
  python run_study.py <config.yaml> train      # teachers, students, robust teachers
  python run_study.py <config.yaml> port       # Acc_A, Acc_scratch, ports, robust ports, drift
  python run_study.py <config.yaml> readouts   # teacher readouts on transfer pool
  python run_study.py <config.yaml> distill    # distillation + labeled-n baselines
  python run_study.py <config.yaml> ablate     # one-factor-at-a-time + oracle + dt + H
  python run_study.py <config.yaml> all

Seeds: device seeds are disjoint by role (teacher / student / robust / pilot /
ablation) and by level; see seed() below.
"""
import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
import src  # noqa: F401
from src.config import load_yaml, HP, Sig, Dyn
from src.data import load
from src.device import devices_from_seeds, threefry_keys
from src.eqprop import init_from_seeds
from src.train import (train_runs, evaluate, accuracy, LABEL, SOFT, HARD)

ROLE = dict(teacher=1, student=2, robust=3, pilot_t=4, pilot_s=5, abl_t=6,
            abl_s=7, init=8, eval=9, distill=10, train=11)


def seed(role, level, i, extra=0):
    return ROLE[role] * 10_000_000 + extra * 100_000 + level * 1000 + i


def rbg_keys(seeds):
    # run-level noise keys (rbg) derived deterministically from integer seeds
    return jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Study:
    def __init__(self, cfg_path):
        self.hp, self.sig, self.dyn, self.cfg = load_yaml(cfg_path)
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        self.out = os.path.join(os.path.dirname(__file__), "results", name)
        os.makedirs(self.out, exist_ok=True)
        xtr, ytr, xte, yte = load(self.cfg.get("dataset", "mnist"))
        self.xtr, self.ytr, self.xte, self.yte = xtr, ytr, xte, yte
        self.X, self.Y = jnp.asarray(xtr), jnp.asarray(ytr)
        self.levels = self.cfg["s_levels"]
        self.U = self.cfg["updates_train"]

    # ------------------------------------------------------------ helpers
    def path(self, *p):
        return os.path.join(self.out, *p)

    def save_params(self, fname, params, meta):
        np.savez_compressed(self.path(fname), **{k: np.asarray(v) for k, v in params.items()},
                            meta=json.dumps(meta))

    def load_params(self, fname):
        d = np.load(self.path(fname))
        return {k: jnp.asarray(d[k]) for k in ("W1", "W2", "bh", "bo")}, json.loads(str(d["meta"]))

    def devs(self, seeds, s, sig=None, dyn=None, hp=None):
        hp = hp or self.hp
        return devices_from_seeds(seeds, s, sig or self.sig, dyn or self.dyn, hp.H)

    def test_acc(self, params, devs, eval_seeds, hp=None, dyn=None, t_ratio=None, dyn_mult=None):
        out = evaluate(params, devs, rbg_keys(eval_seeds), self.xte, hp or self.hp,
                       dyn or self.dyn, t_ratio=t_ratio, dyn_mult=dyn_mult, max_runs=32)
        return accuracy(out, self.yte), out

    def train_group(self, dev_seeds, s, tag, hp=None, sig=None, dyn=None, robust=False,
                    dyn_mult=None):
        hp, sig, dyn = hp or self.hp, sig or self.sig, dyn or self.dyn
        R = len(dev_seeds)
        # init / training-noise seeds are offset copies of the (globally unique)
        # device seed, so every run has its own init and noise stream
        p0 = init_from_seeds([d + 500_000_000 for d in dev_seeds], hp.H, hp)
        keys = rbg_keys([d + 1_000_000_000 for d in dev_seeds])
        if robust:
            devs, dkeys = None, rbg_keys(dev_seeds)
        else:
            devs, dkeys = devices_from_seeds(dev_seeds, s, sig, dyn, hp.H), None
        p, L = train_runs(p0, keys, self.X, self.Y, hp, sig, dyn, self.U, devs=devs,
                          dev_keys=dkeys, s=[s] * R, robust=robust, dyn_mult=dyn_mult,
                          log=log, tag=tag, chunk=self.cfg.get("chunk", 50))
        return p, L

    # ------------------------------------------------------------ stages
    def stage_pilot(self):
        rows = []
        for li, s in enumerate(self.cfg["pilot_levels"]):
            ts = [seed("pilot_t", li, i) for i in range(3)]
            ss = [seed("pilot_s", li, j) for j in range(3)]
            p, L = self.train_group(ts + ss, s, f"pilot s={s}")
            tp = jax.tree_util.tree_map(lambda a: a[:3], p)
            sp = jax.tree_util.tree_map(lambda a: a[3:], p)
            dT, dS = self.devs(ts, s), self.devs(ss, s)
            accA, _ = self.test_acc(tp, dT, [seed("eval", li, i, 50) for i in range(3)])
            accS, _ = self.test_acc(sp, dS, [seed("eval", li, j, 51) for j in range(3)])
            pt = jax.tree_util.tree_map(lambda a: jnp.repeat(a, 3, 0), tp)
            ds = jax.tree_util.tree_map(lambda a: jnp.tile(a, (3,) + (1,) * (a.ndim - 1)), dS)
            accP, _ = self.test_acc(pt, ds, [seed("eval", li, k, 52) for k in range(9)])
            for k, (i, j) in enumerate(itertools.product(range(3), range(3))):
                rows.append(dict(s=s, teacher=i, student=j, acc_A=accA[i],
                                 acc_scratch=accS[j], acc_port=accP[k]))
            log(f"pilot s={s}: accA {accA.round(4)} accS {accS.round(4)} port {accP.round(3)}")
        df = pd.DataFrame(rows)
        df["R"] = (df.acc_port - 0.1) / (df.acc_A - 0.1)
        df.to_parquet(self.path("pilot.parquet"))
        log(df.groupby("s")[["acc_A", "acc_scratch", "acc_port", "R"]].mean().to_string())

    def stage_train(self):
        NT, NS, NR = self.cfg["n_teachers"], self.cfg["n_students"], self.cfg["n_robust"]
        steps = self.hp.langevin_steps_per_update() * self.U
        for li, s in enumerate(self.levels):
            f = f"train_s{li}.npz"
            if not os.path.exists(self.path(f)):
                ts = [seed("teacher", li, i) for i in range(NT)]
                ss = [seed("student", li, j) for j in range(NS)]
                p, L = self.train_group(ts + ss, s, f"train s={s}")
                self.save_params(f, p, dict(s=s, teachers=ts, students=ss, loss=L.tolist(),
                                            langevin_steps_per_run=steps))
            fr = f"robust_s{li}.npz"
            if NR and s > 0 and not os.path.exists(self.path(fr)):
                rs = [seed("robust", li, i) for i in range(NR)]
                p, L = self.train_group(rs, s, f"robust s={s}", robust=True)
                self.save_params(fr, p, dict(s=s, robust=rs, loss=L.tolist(),
                                             langevin_steps_per_run=steps))

    def stage_port(self):
        rows, own_rows = [], []
        t_ratios = self.cfg.get("drift_t_ratios", [1])
        for li, s in enumerate(self.levels):
            p, meta = self.load_params(f"train_s{li}.npz")
            ts, ss = meta["teachers"], meta["students"]
            NT, NS = len(ts), len(ss)
            devs_all = self.devs(ts + ss, s)
            for tr in t_ratios:
                trv = None if tr == 1 else [tr] * (NT + NS)
                acc_own, _ = self.test_acc(p, devs_all, [seed("eval", li, i, 1) for i in range(NT + NS)],
                                           t_ratio=trv)
                for i, d in enumerate(ts + ss):
                    own_rows.append(dict(s=s, level=li, role="teacher" if i < NT else "student",
                                         idx=i if i < NT else i - NT, t_ratio=tr, acc=acc_own[i]))
                # all teacher x student ports
                tp = jax.tree_util.tree_map(lambda a: jnp.repeat(a[:NT], NS, 0), p)
                sd = self.devs(ss, s)
                sd = jax.tree_util.tree_map(lambda a: jnp.tile(a, (NT,) + (1,) * (a.ndim - 1)), sd)
                accP, _ = self.test_acc(tp, sd, [seed("eval", li, k, 2) for k in range(NT * NS)],
                                        t_ratio=None if tr == 1 else [tr] * NT * NS)
                for k, (i, j) in enumerate(itertools.product(range(NT), range(NS))):
                    rows.append(dict(s=s, level=li, kind="insitu", teacher=i, student=j,
                                     t_ratio=tr, acc_port=accP[k]))
            fr = f"robust_s{li}.npz"
            if os.path.exists(self.path(fr)):
                rp, rmeta = self.load_params(fr)
                NR = len(rmeta["robust"])
                rpp = jax.tree_util.tree_map(lambda a: jnp.repeat(a, NS, 0), rp)
                sd = self.devs(ss, s)
                sd = jax.tree_util.tree_map(lambda a: jnp.tile(a, (NR,) + (1,) * (a.ndim - 1)), sd)
                accR, _ = self.test_acc(rpp, sd, [seed("eval", li, k, 3) for k in range(NR * NS)])
                for k, (i, j) in enumerate(itertools.product(range(NR), range(NS))):
                    rows.append(dict(s=s, level=li, kind="robust", teacher=i, student=j,
                                     t_ratio=1, acc_port=accR[k]))
            log(f"port s={s} done")
        pd.DataFrame(rows).to_parquet(self.path("ports.parquet"))
        pd.DataFrame(own_rows).to_parquet(self.path("own.parquet"))

    def stage_readouts(self):
        pool = self.cfg["transfer_pool"]
        for li, s in enumerate(self.levels):
            if s not in self.cfg["distill_levels"] or os.path.exists(self.path(f"readouts_s{li}.npy")):
                continue
            p, meta = self.load_params(f"train_s{li}.npz")
            NT = len(meta["teachers"])
            tp = jax.tree_util.tree_map(lambda a: a[:NT], p)
            out = evaluate(tp, self.devs(meta["teachers"], s),
                           rbg_keys([seed("eval", li, i, 4) for i in range(NT)]),
                           self.xtr[:pool], self.hp, self.dyn, max_runs=16)
            np.save(self.path(f"readouts_s{li}.npy"), out.astype(np.float32))
            log(f"readouts s={s}: teacher train-acc {accuracy(out, self.ytr[:pool]).round(4)}")

    def stage_distill(self):
        c = self.cfg
        Ud, ns, conds = c["updates_distill"], c["n_transfer"], c["distill_conditions"]
        nstud, tpers = c["distill_students"], c["distill_teachers_per_student"]
        perm = jnp.asarray(np.random.default_rng(c["transfer_perm_seed"]).permutation(c["transfer_pool"]))
        X, Y = self.X[:c["transfer_pool"]], self.Y[:c["transfer_pool"]]
        mode_of = dict(soft=SOFT, hard=HARD, label=LABEL)
        for li, s in enumerate(self.levels):
            if s not in c["distill_levels"]:
                continue
            fout = self.path(f"distill_s{li}.parquet")
            if os.path.exists(fout):
                continue
            p, meta = self.load_params(f"train_s{li}.npz")
            ts, ss = meta["teachers"], meta["students"]
            NT = len(ts)
            readouts = jnp.asarray(np.load(self.path(f"readouts_s{li}.npy")))
            runs = []
            assert nstud * tpers * len(ns) * len(conds) < 1000  # seed index space
            for j in range(nstud):
                for r in range(tpers):
                    t = (j + r) % NT  # Latin-style crossing: each teacher used tpers times
                    for n in ns:
                        for cond in conds:
                            mode, init = cond.split("-")
                            runs.append(dict(student=j, teacher=t, n=n, cond=cond, mode=mode, init=init))
            rows = []
            B = c.get("distill_batch_runs", 40)
            for b0 in range(0, len(runs), B):
                rb = runs[b0:b0 + B]
                R = len(rb)
                sdev = self.devs([ss[r["student"]] for r in rb], s)
                rnd = init_from_seeds([seed("distill", li, b0 + k, 7) for k in range(R)], self.hp.H, self.hp)
                tpar = jax.tree_util.tree_map(lambda a: a[jnp.asarray([r["teacher"] for r in rb])], p)
                isport = jnp.asarray([r["init"] == "ported" for r in rb])
                init = jax.tree_util.tree_map(
                    lambda a, b: jnp.where(isport.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), tpar, rnd)
                keys = rbg_keys([seed("distill", li, b0 + k, 8) for k in range(R)])
                pd_, L = train_runs(init, keys, X, Y, self.hp, self.sig, self.dyn, Ud, devs=sdev,
                                    n_avail=[r["n"] for r in rb],
                                    mode=[mode_of[r["mode"]] for r in rb],
                                    tidx=[r["teacher"] for r in rb], perm=perm,
                                    readouts=readouts, log=log, tag=f"distill s={s} [{b0}:{b0+R}]",
                                    chunk=c.get("chunk", 50))
                acc, _ = self.test_acc(pd_, sdev, [seed("eval", li, b0 + k, 5) for k in range(R)])
                for r, a, l in zip(rb, acc, L[:, -1]):
                    rows.append(dict(r, s=s, level=li, acc=a, final_loss=l, updates=Ud,
                                     langevin_steps=Ud * self.hp.langevin_steps_per_update()))
            pd.DataFrame(rows).to_parquet(fout)
            log(f"distill s={s} done")

    def stage_ablate(self):
        """One-factor-at-a-time (13b), oracle rule (13a), dt (13c), H (13d),
        dynamic noise scaling with s (review fix 7). Exploratory."""
        c = self.cfg["ablation"]
        s, NT, NS = c["s"], c["n_teachers"], c["n_students"]
        variants = {
            "baseline": {},
            "no_gain": dict(sig=dict(g=0.0)),
            "no_offset": dict(sig=dict(o=0.0)),
            "no_node_gain": dict(sig=dict(a=0.0, b=0.0)),
            "no_node_offset": dict(sig=dict(c=0.0)),
            "no_T_spread": dict(sig=dict(T=0.0)),
            "only_gain": dict(sig=dict(o=0.0, a=0.0, b=0.0, c=0.0, T=0.0)),
            "only_node": dict(sig=dict(g=0.0, o=0.0, T=0.0)),
            "no_read_noise": dict(dyn=dict(sigma_r=0.0)),
            "no_asymmetry": dict(dyn=dict(alpha=0.0)),
            "quantized_64": dict(dyn=dict(levels=64)),
            "oracle_rule": dict(hp=dict(oracle=True)),
            # static mismatch off: does dynamic (non-device-specific) noise alone
            # produce mortality? (review fix 7) and dynamic noise x4
            "only_dynamic": dict(sig=dict(g=0.0, o=0.0, a=0.0, b=0.0, c=0.0, T=0.0)),
            "dyn_noise_x4": dict(dyn_mult=4.0),
            "dt_0.1": dict(hp=dict(dt=0.1, n_burn_free=50, n_samp_free=20, n_burn_nudge=20,
                                   n_samp_nudge=20, n_eval_burn=75, n_eval_samp=40)),
            "dt_0.4": dict(hp=dict(dt=0.4, n_burn_free=13, n_samp_free=5, n_burn_nudge=5,
                                   n_samp_nudge=5, n_eval_burn=19, n_eval_samp=10)),
            "H_512": dict(hp=dict(H=512)),
        }
        only = c.get("only")
        rows = []
        fout = self.path("ablation.parquet")
        done = pd.read_parquet(fout) if os.path.exists(fout) else None
        for vi, (name, v) in enumerate(variants.items()):
            if only and name not in only:
                continue
            if done is not None and name in set(done.variant):
                continue
            hp = self.hp._replace(**v.get("hp", {}))
            sig = self.sig._replace(**v.get("sig", {}))
            dyn = self.dyn._replace(**v.get("dyn", {}))
            dm = v.get("dyn_mult", 1.0)
            ts = [seed("abl_t", vi, i) for i in range(NT)]
            ss = [seed("abl_s", vi, j) for j in range(NS)]
            p, L = self.train_group(ts + ss, s, f"ablate {name}", hp=hp, sig=sig, dyn=dyn,
                                    dyn_mult=[dm] * (NT + NS))
            dv = self.devs(ts + ss, s, sig, dyn, hp)
            acc_own, _ = self.test_acc(p, dv, [seed("eval", vi, i, 20) for i in range(NT + NS)],
                                       hp=hp, dyn=dyn, dyn_mult=[dm] * (NT + NS))
            tp = jax.tree_util.tree_map(lambda a: jnp.repeat(a[:NT], NS, 0), p)
            sd = self.devs(ss, s, sig, dyn, hp)
            sd = jax.tree_util.tree_map(lambda a: jnp.tile(a, (NT,) + (1,) * (a.ndim - 1)), sd)
            accP, _ = self.test_acc(tp, sd, [seed("eval", vi, k, 21) for k in range(NT * NS)],
                                    hp=hp, dyn=dyn, dyn_mult=[dm] * NT * NS)
            new = []
            for k, (i, j) in enumerate(itertools.product(range(NT), range(NS))):
                new.append(dict(variant=name, s=s, teacher=i, student=j, acc_A=acc_own[i],
                                acc_scratch=acc_own[NT + j], acc_port=accP[k],
                                langevin_steps=hp.langevin_steps_per_update() * self.U))
            nd = pd.DataFrame(new)
            done = nd if done is None else pd.concat([done, nd])
            done.to_parquet(fout)
            R = ((nd.acc_port - .1) / (nd.acc_A - .1)).mean()
            log(f"ablate {name}: accA {acc_own[:NT].mean():.4f} R {R:.3f}")


if __name__ == "__main__":
    cfg, *stages = sys.argv[1:]
    st = Study(cfg)
    order = ["train", "port", "readouts", "distill", "ablate"]
    for stage in (order if stages == ["all"] else stages):
        log(f"=== stage {stage} ===")
        getattr(st, f"stage_{stage}")()
