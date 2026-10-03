"""Regenerate every figure and table from results/<cfg>/*.parquet.

  python make_report.py configs/main.yaml [configs/fashion.yaml]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.config import load_yaml
from src import analysis as A

NB = 10000
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = dict(R_A="#1f4e79", R_B="#2e8b57", robust="#c0504d", soft_random="#1f4e79",
         hard_random="#7f7f7f", label_random="#d4a017", soft_ported="#2e8b57",
         label_ported="#c0504d")


def sx(s):
    """x position: s=0 drawn at a fixed point left of the smallest s>0."""
    s = np.asarray(s, float)
    pos = s[s > 0]
    left = pos.min() / 2.5 if len(pos) else 0.1
    return np.where(s > 0, s, left)


def fmt_ci(x, lo, hi, d=3):
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def mortality(res, out, hp, title, lines):
    ports = pd.read_parquet(os.path.join(res, "ports.parquet"))
    own = pd.read_parquet(os.path.join(res, "own.parquet"))
    mats = A.pair_matrices(ports, own)
    MA = A.mortality_bootstrap(mats, "R_A", NB, seed=1)
    MB = A.mortality_bootstrap(mats, "R_B", NB, seed=2)
    M2 = A.mortality_bootstrap(mats, "R_A", 2000, seed=3, floor=False)
    rob = A.pair_matrices(ports, own, kind="robust") if (ports.kind == "robust").any() else {}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sgrid = np.geomspace(min(MA["levels"][MA["levels"] > 0]) / 2.5, max(MA["levels"]) * 1.5, 200)
    for M, key, lab in [(MA, "R_A", r"$R_A$ (vs teacher)"), (MB, "R_B", r"$R_B$ (vs student scratch)")]:
        lo, hi = np.nanpercentile(M["boots"], [2.5, 97.5], axis=0)
        x = sx(M["levels"])
        ax.errorbar(x, M["point"], yerr=[M["point"] - lo, hi - M["point"]], fmt="o", ms=4,
                    color=C[key], label=lab, capsize=2)
        band = np.array([A.hill(sgrid, *f) for f in M["fits"] if np.all(np.isfinite(f))])
        ax.plot(sgrid, A.hill(sgrid, *M["fit"]), color=C[key], lw=1.2)
        ax.fill_between(sgrid, *np.nanpercentile(band, [2.5, 97.5], 0), color=C[key], alpha=0.15, lw=0)
    if rob:
        rl = sorted(rob)
        rp = [np.nanmean(rob[s]["R_B"]) for s in rl]
        rci = np.array([A.ci(A.two_way_boot_means(rob[s]["R_B"], 2000, np.random.default_rng(4))) for s in rl])
        ax.errorbar(sx(rl), rp, yerr=[np.array(rp) - rci[:, 0], rci[:, 1] - np.array(rp)], fmt="s",
                    ms=4, color=C["robust"], label=r"robust-once-and-copy ($R_B$)", capsize=2)
    ax.set_xscale("log")
    ax.axhline(0.5, ls=":", c="k", lw=0.7)
    ax.set_xlabel("mismatch scale s  (device area = 1/s² µm²)")
    ax.set_ylabel("retention R")
    ax.set_title(title)
    sec = ax.secondary_xaxis("top", functions=(lambda s: 1 / np.maximum(s, 1e-9) ** 2,
                                               lambda a: 1 / np.sqrt(np.maximum(a, 1e-9))))
    sec.set_xlabel("W·L (µm²)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "fig1_mortality_curve.png"))
    fig.savefig(os.path.join(out, "fig1_mortality_curve.pdf"))
    plt.close(fig)

    lines.append(f"## Mortality curve ({title})\n")
    tab = []
    for i, s in enumerate(MA["levels"]):
        m = mats[s]
        lo, hi = A.ci(MA["boots"][:, i])
        lb, hb = A.ci(MB["boots"][:, i])
        row = dict(s=s, area_um2=(1 / s ** 2 if s > 0 else np.inf),
                   acc_A=np.mean(m["A"]), acc_scratch=np.mean(m["S"]), acc_port=np.nanmean(m["port"]),
                   R_A=fmt_ci(MA["point"][i], lo, hi), R_B=fmt_ci(MB["point"][i], lb, hb))
        if s in rob:
            row["acc_robust_port"] = np.nanmean(rob[s]["port"])
        row["min_acc_own"] = min(np.min(m["A"]), np.min(m["S"]))
        tab.append(row)
    lines.append(pd.DataFrame(tab).to_markdown(index=False, floatfmt=".4f"))
    for M, nm in [(MA, "R_A, 3-par"), (MB, "R_B, 3-par"), (M2, "R_A, 2-par")]:
        lo, hi = A.ci(M["s_half_boot"])
        s50, k, rinf = M["fit"]
        lines.append(f"\n- Hill fit ({nm}): s50={s50:.3f}, k={k:.2f}, R_inf={rinf:.3f}; "
                     f"**s_half = {M['s_half']:.3f} [{lo:.3f}, {hi:.3f}]** "
                     f"(area {1/M['s_half']**2:.3f} µm²); "
                     f"fits failed in {np.mean(~np.isfinite(M['s_half_boot']))*100:.1f}% of replicates")
    lines.append("")
    return mats, MA, ports, own


def distillation(res, out, mats, ports, lines):
    files = sorted(f for f in os.listdir(res) if f.startswith("distill_s"))
    if not files:
        return None
    dist = pd.concat([pd.read_parquet(os.path.join(res, f)) for f in files])
    T = A.distill_table(dist, mats)
    T.to_parquet(os.path.join(out, "distill_table.parquet"))
    levels = sorted(T.s.unique())
    conds = list(dict.fromkeys(T.cond))
    rob = ports[ports.kind == "robust"]
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, levels):
        g = T[T.s == s]
        for cnd in conds:
            m = g[g.cond == cnd].groupby("n").acc
            mean = m.mean()
            boot = A.student_boot(g[g.cond == cnd], "acc",
                                  lambda d: d.groupby("n").acc.mean().to_numpy(), 1000, seed=5)
            lo, hi = np.percentile(boot, [2.5, 97.5], 0)
            ax.plot(mean.index, mean.values, "-o", ms=3, color=C[cnd.replace("-", "_")], label=cnd)
            ax.fill_between(mean.index, lo, hi, color=C[cnd.replace("-", "_")], alpha=0.12, lw=0)
        ax.axhline(np.nanmean(mats[s]["port"]), color="k", ls="--", lw=0.8, label="port (no retraining)")
        ax.axhline(np.mean(mats[s]["S"]), color="k", ls=":", lw=0.8, label="scratch (full budget)")
        rs = rob[rob.s == s]
        if len(rs):
            ax.axhline(rs.acc_port.mean(), color=C["robust"], ls="-.", lw=0.8, label="robust-once-and-copy")
        ax.set_xscale("log")
        ax.set_title(f"s = {s}  (W·L = {1/s**2:.3g} µm²)")
        ax.set_xlabel("transfer-set size n")
    axes[0].set_ylabel("student test accuracy")
    axes[-1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(os.path.join(out, "fig2_distillation.png"))
    fig.savefig(os.path.join(out, "fig2_distillation.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for s in levels:
        g = T[(T.s == s) & (T.cond == "soft-random")].groupby("n").D.mean()
        ax.plot(g.index, g.values, "-o", ms=3, label=f"s={s}")
    ax.set_xscale("log"); ax.axhline(0, c="k", lw=0.6); ax.axhline(1, c="k", lw=0.6, ls=":")
    ax.set_xlabel("n"); ax.set_ylabel("recovery D(n), soft targets, random init")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "fig2b_recovery.png")); plt.close(fig)

    lines.append("## Distillation\n")
    piv = T.pivot_table(index=["s", "n"], columns="cond", values="acc", aggfunc="mean")
    lines.append(piv.to_markdown(floatfmt=".4f"))
    pivD = T.pivot_table(index=["s", "n"], columns="cond", values="D", aggfunc="mean")
    lines.append("\nRecovery D(n):\n")
    lines.append(pivD.to_markdown(floatfmt=".3f"))
    lines.append("")
    return T


def hypotheses(MA, T, ports, lines, out):
    res = []
    lv = MA["levels"]
    sh_lo, sh_hi = A.ci(MA["s_half_boot"])
    in_range = (sh_lo >= lv[lv > 0].min()) and (sh_hi <= lv.max())
    res.append(dict(id="H1", estimate=f"rho={MA['rho_point']:.3f}; s_half={MA['s_half']:.3f} [{sh_lo:.3f},{sh_hi:.3f}]",
                    p=MA["p_H1"], extra_condition=f"s_half CI inside tested range: {in_range}"))
    if T is not None:
        sr = T[T.cond == "soft-random"].dropna(subset=["D"])
        b = A.student_boot(sr, "D", lambda d: A.slope_logn(d), NB, seed=6)
        est = A.slope_logn(sr)
        inc = sr.groupby("n").D.mean().diff().dropna().round(3).tolist()
        res.append(dict(id="H2", estimate=f"slope dD/dlog10n={est:.3f} CI {A.ci(b).round(3).tolist()}; increments {inc}",
                        p=float(np.mean(b <= 0)), extra_condition=""))
        sub = T[(T.n == 100) & T.cond.isin(["soft-ported", "soft-random"])]
        f = lambda d: A.paired_diff(d, "soft-ported", "soft-random")
        b = A.student_boot(sub, "acc", f, NB, seed=7)
        res.append(dict(id="H3", estimate=f"ported-random acc @n=100: {f(sub):.4f} CI {A.ci(b).round(4).tolist()}",
                        p=float(np.mean(b <= 0)), extra_condition=""))
    adj = A.holm([r["p"] for r in res])
    for r, a in zip(res, adj):
        r["p_holm"] = a
        r["pass"] = bool(a < 0.05 and (r["id"] != "H1" or "True" in r["extra_condition"]))
    sec = []
    if T is not None:
        sub = T[(T.n <= 3000) & T.cond.isin(["soft-random", "hard-random"])]
        f = lambda d: A.paired_diff(d, "soft-random", "hard-random")
        b = A.student_boot(sub, "acc", f, NB, seed=8)
        sec.append(dict(id="S1 soft>hard", estimate=f"{f(sub):.4f} CI {A.ci(b).round(4).tolist()}",
                        p=float(np.mean(b <= 0))))
        sub = T[T.cond.isin(["soft-random", "label-random"])]
        f = lambda d: A.paired_diff(d, "soft-random", "label-random")
        b = A.student_boot(sub, "acc", f, NB, seed=9)
        sec.append(dict(id="S2 soft vs labeled-n (two-sided)", estimate=f"{f(sub):.4f} CI {A.ci(b).round(4).tolist()}",
                        p=float(2 * min(np.mean(b <= 0), np.mean(b >= 0)))))
    rob = ports[(ports.kind == "robust")]
    ins = ports[(ports.kind == "insitu") & (ports.t_ratio == 1) & (ports.s > 0)]
    if len(rob):
        rng = np.random.default_rng(10)
        diffs = []
        for _ in range(NB):
            tot = []
            for s in sorted(rob.s.unique()):
                r, i = rob[rob.s == s], ins[ins.s == s]
                js = rng.choice(r.student.unique(), r.student.nunique())
                rt = rng.choice(r.teacher.unique(), r.teacher.nunique())
                it = rng.choice(i.teacher.unique(), i.teacher.nunique())
                ra = r.set_index(["teacher", "student"]).acc_port
                ia = i.set_index(["teacher", "student"]).acc_port
                tot.append(np.mean([ra[(t, j)] for t in rt for j in js]) - np.mean([ia[(t, j)] for t in it for j in js]))
            diffs.append(np.mean(tot))
            if _ >= 1999:
                break
        est = np.mean([rob[rob.s == s].acc_port.mean() - ins[ins.s == s].acc_port.mean() for s in rob.s.unique()])
        sec.append(dict(id="S3 robust port > in-situ port", estimate=f"{est:.4f} CI {A.ci(diffs).round(4).tolist()}",
                        p=float(np.mean(np.array(diffs) <= 0))))
    if sec:
        adj = A.holm([r["p"] for r in sec])
        for r, a in zip(sec, adj):
            r["p_holm"] = a
            r["pass"] = bool(a < 0.05)
    lines.append("## Table 3: hypothesis outcomes\n")
    lines.append(pd.DataFrame(res).to_markdown(index=False, floatfmt=".4g"))
    if sec:
        lines.append("\nSecondary (pre-registered):\n")
        lines.append(pd.DataFrame(sec).to_markdown(index=False, floatfmt=".4g"))
    lines.append("")
    if T is not None:
        lines.append("Exploratory crossovers:\n")
        rows = []
        for s in sorted(T.s.unique()):
            sub = T[(T.s == s) & (T.n == 100) & T.cond.isin(["soft-ported", "soft-random"])]
            f = lambda d: A.paired_diff(d, "soft-ported", "soft-random")
            b = A.student_boot(sub, "acc", f, 2000, seed=11)
            sp = T[(T.s == s) & (T.n == 1000) & (T.cond == "soft-ported")].acc.mean()
            rp = rob[rob.s == s].acc_port.mean() if len(rob) else np.nan
            rows.append(dict(s=s, ported_minus_random_n100=fmt_ci(f(sub), *A.ci(b), d=4),
                             distill_soft_ported_n1000=sp, robust_port=rp))
        lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
        lines.append("")


def drift(own, ports, mats, out, lines):
    trs = sorted(ports.t_ratio.unique())
    if len(trs) < 2:
        return
    rows = []
    for tr in trs:
        m = A.pair_matrices(ports, own, t_ratio=tr)
        for s in sorted(m):
            rows.append(dict(t_ratio=tr, s=s, acc_A=np.mean(m[s]["A"]), acc_port=np.nanmean(m[s]["port"]),
                             R_A=np.nanmean(m[s]["R_A"])))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4, 3))
    for s, g in D.groupby("s"):
        ax.plot(g.t_ratio, g.R_A, "-o", ms=3, label=f"s={s}")
    ax.set_xscale("log"); ax.set_xlabel("t / t0"); ax.set_ylabel("R_A (both devices drifted)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(os.path.join(out, "fig6_drift.png")); plt.close(fig)
    lines.append("## H4 (exploratory): drift\n")
    lines.append(D.pivot_table(index="s", columns="t_ratio", values=["acc_A", "R_A"]).to_markdown(floatfmt=".4f"))
    lines.append("")


def ablation(res, out, lines):
    f = os.path.join(res, "ablation.parquet")
    if not os.path.exists(f):
        return
    ab = pd.read_parquet(f)
    rows = []
    for v, g in ab.groupby("variant", sort=False):
        NT, NS = g.teacher.max() + 1, g.student.max() + 1
        M = np.full((NT, NS), np.nan)
        M[g.teacher, g.student] = ((g.acc_port - .1) / (g.acc_A - .1)).to_numpy()
        b = A.two_way_boot_means(M, 4000, np.random.default_rng(12))
        rows.append(dict(variant=v, acc_A=g.groupby("teacher").acc_A.first().mean(),
                         acc_scratch=g.groupby("student").acc_scratch.first().mean(),
                         acc_port=g.acc_port.mean(), mu=1 - np.nanmean(M),
                         mu_lo=1 - np.percentile(b, 97.5), mu_hi=1 - np.percentile(b, 2.5),
                         langevin_steps=g.langevin_steps.iloc[0]))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    y = np.arange(len(D))
    ax.barh(y, D.mu, xerr=[D.mu - D.mu_lo, D.mu_hi - D.mu], color="#1f4e79", alpha=0.8, capsize=2)
    ax.set_yticks(y, D.variant); ax.invert_yaxis()
    ax.axvline(D[D.variant == "baseline"].mu.iloc[0], c="k", ls=":", lw=0.8)
    ax.set_xlabel(f"mortality index mu at s = {ab.s.iloc[0]}")
    fig.savefig(os.path.join(out, "fig4_ablation.png")); fig.savefig(os.path.join(out, "fig4_ablation.pdf"))
    plt.close(fig)
    lines.append(f"## Ablations (exploratory, s = {ab.s.iloc[0]})\n")
    lines.append(D.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")


def main(cfgs):
    for cfg in cfgs:
        hp, sig, dyn, d = load_yaml(cfg)
        name = os.path.splitext(os.path.basename(cfg))[0]
        res = os.path.join(os.path.dirname(__file__), "results", name)
        out = os.path.join(res, "report")
        os.makedirs(out, exist_ok=True)
        lines = [f"# Study 6 results: {name} ({d.get('dataset', 'mnist')})\n",
                 "Simulation, Level 1 (calibrated device statistics). Auto-generated by make_report.py.\n"]
        lines.append("## Table 1: device-model baselines\n")
        lines.append(A.calibration_table(sig, dyn, hp).to_markdown(index=False))
        lines.append("")
        mats, MA, ports, own = mortality(res, out, hp, d.get("dataset", "mnist"), lines)
        T = distillation(res, out, mats, ports, lines)
        hypotheses(MA, T, ports, lines, out)
        drift(own, ports, mats, out, lines)
        ablation(res, out, lines)
        steps = hp.langevin_steps_per_update()
        lines.append("## Table 2: compute (Langevin steps per run, all phases, all samples)\n")
        comp = [dict(method="teacher / scratch training", updates=d["updates_train"],
                     langevin_steps=d["updates_train"] * steps * hp.batch),
                dict(method="robust-once-and-copy teacher", updates=d["updates_train"],
                     langevin_steps=d["updates_train"] * steps * hp.batch)]
        if T is not None:
            comp.append(dict(method="distillation / labeled-n (any n)", updates=d["updates_distill"],
                             langevin_steps=d["updates_distill"] * steps * hp.batch))
        comp.append(dict(method="port (inference only)", updates=0, langevin_steps=0))
        lines.append(pd.DataFrame(comp).to_markdown(index=False))
        lines.append("\n(Langevin steps counted per sample: free + both nudged phases, burn-in + samples.)\n")
        with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main(sys.argv[1:])
