"""Calibration sensitivity: how does the mortality curve move with the coupling
offset spread sigma_o (the dominant driver in the ablation, and an assumed value)?

sigma_o is in units of w_max, so varying it also varies the absolute offset scale
that the tuned weight bound implies. Frozen hyperparameters; fresh device seeds.
The s = 0 anchor (no static mismatch, sigma_o irrelevant) and the baseline
sigma_o = 0.01 curve come from the main study.

  python run_sensitivity.py configs/main.yaml          # run (resumable)
  python run_sensitivity.py configs/main.yaml report   # analysis only
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import run_study as RS
from src import analysis as A

SIGMA_O = [0.0025, 0.005, 0.02]
LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
NT = NS = 4
RS.ROLE.update(sens_t=12, sens_s=13)


def run(st):
    fout = st.path("sensitivity.parquet")
    done = pd.read_parquet(fout) if os.path.exists(fout) else None
    for oi, so in enumerate(SIGMA_O):
        sig = st.sig._replace(o=so)
        for li, s in enumerate(LEVELS):
            if done is not None and ((done.sigma_o == so) & (done.s == s)).any():
                continue
            ts = [RS.seed("sens_t", li, i, oi) for i in range(NT)]
            ss = [RS.seed("sens_s", li, j, oi) for j in range(NS)]
            p, _ = st.train_group(ts + ss, s, f"sens sigma_o={so} s={s}", sig=sig)
            dv = st.devs(ts + ss, s, sig)
            acc_own, _ = st.test_acc(p, dv, [RS.seed("eval", li, i, 30 + oi) for i in range(NT + NS)])
            tp = jax.tree_util.tree_map(lambda a: jnp.repeat(a[:NT], NS, 0), p)
            sd = st.devs(ss, s, sig)
            sd = jax.tree_util.tree_map(lambda a: jnp.tile(a, (NT,) + (1,) * (a.ndim - 1)), sd)
            accP, _ = st.test_acc(tp, sd, [RS.seed("eval", li, k, 40 + oi) for k in range(NT * NS)])
            new = pd.DataFrame([dict(sigma_o=so, s=s, teacher=i, student=j, acc_A=acc_own[i],
                                     acc_scratch=acc_own[NT + j], acc_port=accP[k])
                                for k, (i, j) in enumerate(itertools.product(range(NT), range(NS)))])
            done = new if done is None else pd.concat([done, new])
            done.to_parquet(fout)
            R = ((new.acc_port - .1) / (new.acc_A - .1)).mean()
            RS.log(f"sens sigma_o={so} s={s}: accA {acc_own[:NT].mean():.4f} "
                   f"accS {acc_own[NT:].mean():.4f} port {accP.mean():.4f} R {R:.3f}")


def report(st):
    sens = pd.read_parquet(st.path("sensitivity.parquet"))
    ports = pd.read_parquet(st.path("ports.parquet"))
    own = pd.read_parquet(st.path("own.parquet"))
    main = A.pair_matrices(ports, own)
    curves = {}
    for so, g in sens.groupby("sigma_o"):
        mats = {0.0: main[0.0]}  # anchor: s = 0 has no offsets at all
        for s, h in g.groupby("s"):
            M = np.full((NT, NS), np.nan)
            M[h.teacher, h.student] = h.acc_port
            Av = h.groupby("teacher").acc_A.first().to_numpy()
            Sv = h.groupby("student").acc_scratch.first().to_numpy()
            mats[s] = dict(port=M, A=Av, S=Sv, R_A=(M - .1) / (Av[:, None] - .1),
                           R_B=(M - .1) / (Sv[None, :] - .1))
        curves[so] = (mats, A.mortality_bootstrap(mats, "R_A", 4000, seed=21))
    base = {s: main[s] for s in main}
    curves[0.01] = (base, A.mortality_bootstrap(base, "R_A", 4000, seed=22))

    out = st.path("report")
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    cols = {0.0025: "#9ecae1", 0.005: "#4292c6", 0.01: "#1f4e79", 0.02: "#c0504d"}
    grid = np.geomspace(0.1, 5, 200)
    rows = []
    for so in sorted(curves):
        mats, M = curves[so]
        lv = M["levels"]
        lo, hi = np.nanpercentile(M["boots"], [2.5, 97.5], 0)
        x = np.where(lv > 0, lv, 0.1)
        ax.errorbar(x, M["point"], yerr=[M["point"] - lo, hi - M["point"]], fmt="o", ms=3.5,
                    color=cols[so], capsize=2, label=f"σ_o = {so}" + (" (baseline)" if so == 0.01 else ""))
        ax.plot(grid, A.hill(grid, *M["fit"]), color=cols[so], lw=1.1)
        shlo, shhi = A.ci(M["s_half_boot"])
        min_own = min(min(np.min(mats[s]["A"]), np.min(mats[s]["S"])) for s in mats)
        rows.append(dict(sigma_o=so, s_half=M["s_half"], s_half_lo=shlo, s_half_hi=shhi,
                         area_um2=1 / M["s_half"] ** 2, k=M["fit"][1], R_inf=M["fit"][2],
                         **{f"R(s={s:g})": M["point"][i] for i, s in enumerate(lv) if s in (0.5, 1.0, 2.0)},
                         min_own_acc=min_own))
    ax.set_xscale("log"); ax.axhline(0.5, c="k", ls=":", lw=0.7)
    ax.set_xlabel("mismatch scale s"); ax.set_ylabel("retention $R_A$")
    ax.set_title("Sensitivity to the coupling-offset spread σ_o")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "fig7_sigma_o_sensitivity.png"))
    fig.savefig(os.path.join(out, "fig7_sigma_o_sensitivity.pdf"))
    plt.close(fig)
    T = pd.DataFrame(rows)
    # elasticity of s_half w.r.t. sigma_o (log-log slope)
    slope = np.polyfit(np.log(T.sigma_o), np.log(T.s_half), 1)[0]
    lines = ["# Calibration sensitivity: coupling-offset spread σ_o\n",
             f"Levels s ∈ {LEVELS}, {NT}+{NS} devices per cell (σ_o ≠ 0.01); baseline curve and s = 0 "
             "anchor from the main study. 4000 two-way cluster-bootstrap replicates.\n",
             T.to_markdown(index=False, floatfmt=".3f"),
             f"\nlog-log slope d ln s_half / d ln σ_o = {slope:.3f} "
             f"(0 = insensitive; −1 = s_half inversely proportional to σ_o)\n"]
    with open(os.path.join(out, "sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    st = RS.Study(sys.argv[1])
    if len(sys.argv) < 3 or sys.argv[2] != "report":
        run(st)
    report(st)
