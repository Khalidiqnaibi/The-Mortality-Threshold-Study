"""Does the distillation-vs-robust-copy ordering survive the calibration
uncertainty in sigma_o? Re-runs a reduced version of the port / robust /
distillation comparison at the two extreme offset spreads of the sensitivity
check, at mismatch levels bracketing each one's own cliff.

  python run_sens_distill.py configs/main.yaml          # run (resumable)
  python run_sens_distill.py configs/main.yaml report   # analysis only
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
import run_study as RS
from src.eqprop import init_from_seeds
from src.train import train_runs, evaluate, accuracy, SOFT
from src import analysis as A

CELLS = {0.02: [0.5, 1.0, 1.5], 0.0025: [1.0, 1.5, 2.0]}
NT = NS = 4
NR = 3
TPS = 2                      # teachers per student (Latin crossing)
NS_TRANSFER = [100, 1000]
CONDS = ["soft-ported", "soft-random"]
RS.ROLE.update(sd_t=14, sd_s=15, sd_r=16, sd_d=17)


def cell(st, so, oi, s, li):
    sig = st.sig._replace(o=so)
    ex = 10 * oi + li
    ts = [RS.seed("sd_t", li, i, oi) for i in range(NT)]
    ss = [RS.seed("sd_s", li, j, oi) for j in range(NS)]
    rs = [RS.seed("sd_r", li, r, oi) for r in range(NR)]
    tag = f"sigma_o={so} s={s}"
    p, _ = st.train_group(ts + ss, s, f"{tag} insitu", sig=sig)
    rp, _ = st.train_group(rs, s, f"{tag} robust", sig=sig, robust=True)
    dv = st.devs(ts + ss, s, sig)
    acc_own, _ = st.test_acc(p, dv, [RS.seed("eval", li, i, 60 + oi) for i in range(NT + NS)])
    sd = st.devs(ss, s, sig)
    tile = lambda d, k: jax.tree_util.tree_map(lambda a: jnp.tile(a, (k,) + (1,) * (a.ndim - 1)), d)
    tp = jax.tree_util.tree_map(lambda a: jnp.repeat(a[:NT], NS, 0), p)
    accP, _ = st.test_acc(tp, tile(sd, NT), [RS.seed("eval", li, k, 62 + oi) for k in range(NT * NS)])
    rpp = jax.tree_util.tree_map(lambda a: jnp.repeat(a, NS, 0), rp)
    accR, _ = st.test_acc(rpp, tile(sd, NR), [RS.seed("eval", li, k, 64 + oi) for k in range(NR * NS)])
    rows = []
    for k, (i, j) in enumerate(itertools.product(range(NT), range(NS))):
        rows.append(dict(kind="port", teacher=i, student=j, n=0, acc=accP[k]))
    for k, (i, j) in enumerate(itertools.product(range(NR), range(NS))):
        rows.append(dict(kind="robust_port", teacher=i, student=j, n=0, acc=accR[k]))
    for j in range(NS):
        rows.append(dict(kind="scratch", teacher=-1, student=j, n=0, acc=acc_own[NT + j]))
    for i in range(NT):
        rows.append(dict(kind="teacher_own", teacher=i, student=-1, n=0, acc=acc_own[i]))

    # teacher readouts on the transfer pool, then output-only distillation
    pool = st.cfg["transfer_pool"]
    tparams = jax.tree_util.tree_map(lambda a: a[:NT], p)
    readouts = jnp.asarray(evaluate(tparams, st.devs(ts, s, sig),
                                    RS.rbg_keys([RS.seed("eval", li, i, 66 + oi) for i in range(NT)]),
                                    st.xtr[:pool], st.hp, st.dyn, max_runs=16).astype(np.float32))
    perm = jnp.asarray(np.random.default_rng(st.cfg["transfer_perm_seed"]).permutation(pool))
    runs = [dict(student=j, teacher=(j + r) % NT, n=n, cond=c)
            for j in range(NS) for r in range(TPS) for n in NS_TRANSFER for c in CONDS]
    R = len(runs)
    sdev = st.devs([ss[r["student"]] for r in runs], s, sig)
    rnd = init_from_seeds([RS.seed("sd_d", li, k, oi) for k in range(R)], st.hp.H, st.hp)
    tpar = jax.tree_util.tree_map(lambda a: a[jnp.asarray([r["teacher"] for r in runs])], p)
    isport = jnp.asarray([r["cond"] == "soft-ported" for r in runs])
    init = jax.tree_util.tree_map(
        lambda a, b: jnp.where(isport.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), tpar, rnd)
    pd_, _ = train_runs(init, RS.rbg_keys([RS.seed("sd_d", li, k, 50 + oi) for k in range(R)]),
                        st.X[:pool], st.Y[:pool], st.hp, sig, st.dyn, st.cfg["updates_distill"],
                        devs=sdev, n_avail=[r["n"] for r in runs], mode=[SOFT] * R,
                        tidx=[r["teacher"] for r in runs], perm=perm, readouts=readouts,
                        log=RS.log, tag=f"{tag} distill")
    accD, _ = st.test_acc(pd_, sdev, [RS.seed("eval", li, k, 68 + oi) for k in range(R)])
    for r, a in zip(runs, accD):
        rows.append(dict(kind=r["cond"], teacher=r["teacher"], student=r["student"], n=r["n"], acc=a))
    df = pd.DataFrame(rows)
    df["sigma_o"], df["s"] = so, s
    return df


def run(st):
    fout = st.path("sens_distill.parquet")
    done = pd.read_parquet(fout) if os.path.exists(fout) else None
    for oi, (so, levels) in enumerate(CELLS.items()):
        for li, s in enumerate(levels):
            if done is not None and ((done.sigma_o == so) & (done.s == s)).any():
                continue
            df = cell(st, so, oi, s, li)
            done = df if done is None else pd.concat([done, df])
            done.to_parquet(fout)
            g = df.groupby(["kind", "n"]).acc.mean()
            RS.log(f"sigma_o={so} s={s}: " + ", ".join(f"{k}{'' if n == 0 else f'@{n}'}={v:.3f}"
                                                         for (k, n), v in g.items()))


def report(st):
    d = pd.read_parquet(st.path("sens_distill.parquet"))
    # baseline sigma_o = 0.01 from the main study, same summaries
    ports = pd.read_parquet(st.path("ports.parquet"))
    own = pd.read_parquet(st.path("own.parquet"))
    dist = pd.concat([pd.read_parquet(st.path(f)) for f in sorted(os.listdir(st.out))
                      if f.startswith("distill_s")])
    base = []
    for s in sorted(dist.s.unique()):
        P = ports[(ports.s == s) & (ports.t_ratio == 1)]
        row = dict(sigma_o=0.01, s=s, port=P[P.kind == "insitu"].acc_port.mean(),
                   robust_port=P[P.kind == "robust"].acc_port.mean(),
                   scratch=own[(own.s == s) & (own.t_ratio == 1) & (own.role == "student")].acc.mean())
        for c in CONDS:
            for n in NS_TRANSFER:
                row[f"{c}@{n}"] = dist[(dist.s == s) & (dist.cond == c) & (dist.n == n)].acc.mean()
        base.append(row)
    rows = []
    rng = np.random.default_rng(31)
    for (so, s), g in d.groupby(["sigma_o", "s"]):
        row = dict(sigma_o=so, s=s, port=g[g.kind == "port"].acc.mean(),
                   robust_port=g[g.kind == "robust_port"].acc.mean(),
                   scratch=g[g.kind == "scratch"].acc.mean())
        for c in CONDS:
            for n in NS_TRANSFER:
                row[f"{c}@{n}"] = g[(g.kind == c) & (g.n == n)].acc.mean()
        # bootstrap over students: distill(soft-ported, n=100) - robust port
        rb = g[g.kind == "robust_port"].groupby("student").acc.mean()
        ds = g[(g.kind == "soft-ported") & (g.n == 100)].groupby("student").acc.mean()
        diff = (ds - rb).to_numpy()
        b = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
        row["distill100_minus_robust"] = f"{diff.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
        rows.append(row)
    T = pd.concat([pd.DataFrame(base), pd.DataFrame(rows)]).sort_values(["sigma_o", "s"])
    out = st.path("report")
    os.makedirs(out, exist_ok=True)
    txt = ["# Distillation vs robust-copy under sigma_o calibration uncertainty\n",
           f"{NT}+{NS} devices, {NR} robust teachers per cell; distillation: {NS}x{TPS} pairs, "
           f"soft targets, {st.cfg['updates_distill']} updates. sigma_o = 0.01 rows are from the main study.\n",
           T.to_markdown(index=False, floatfmt=".3f")]
    with open(os.path.join(out, "sens_distill.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print("\n".join(txt).encode("ascii", "replace").decode())


if __name__ == "__main__":
    st = RS.Study(sys.argv[1])
    if len(sys.argv) < 3 or sys.argv[2] != "report":
        run(st)
    report(st)
