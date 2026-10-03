"""Statistics for Study 6 (protocol Section 10, with review fixes).

Units of replication are devices. Mortality uses a two-way cluster bootstrap
(teacher IDs and student IDs resampled independently, within each s level).
Distillation uses a student-cluster bootstrap (each student is crossed with
`distill_teachers_per_student` teachers in a Latin design).
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

FLOOR = 0.1


# ---------------------------------------------------------------- Hill curve
def hill(s, s50, k, rinf):
    s = np.asarray(s, float)
    return rinf + (1 - rinf) / (1 + (np.maximum(s, 0) / s50) ** k)


def s_half(s50, k, rinf):
    """Mismatch at which R = 0.5 (the single-number mortality scale)."""
    if rinf >= 0.5:
        return np.inf
    return s50 * ((1 - rinf) / (0.5 - rinf) - 1) ** (1 / k)


def fit_hill(s, R, floor=True):
    s, R = np.asarray(s, float), np.asarray(R, float)
    try:
        if floor:
            p, _ = curve_fit(hill, s, R, p0=[np.median(s[s > 0]), 2.0, 0.0],
                             bounds=([1e-3, 0.2, -0.5], [1e3, 12, 0.99]), maxfev=20000)
        else:
            f = lambda s, s50, k: hill(s, s50, k, 0.0)
            p, _ = curve_fit(f, s, R, p0=[np.median(s[s > 0]), 2.0],
                             bounds=([1e-3, 0.2], [1e3, 12]), maxfev=20000)
            p = np.r_[p, 0.0]
        return p
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


# ---------------------------------------------------------------- mortality
def pair_matrices(ports, own, t_ratio=1, kind="insitu"):
    """Per level: R_A (normalised by teacher), R_B (normalised by student
    scratch accuracy), port accuracy matrices [teacher, student]."""
    out = {}
    P = ports[(ports.kind == kind) & (ports.t_ratio == t_ratio)]
    O = own[own.t_ratio == t_ratio]
    for s, g in P.groupby("s"):
        NT, NS = g.teacher.max() + 1, g.student.max() + 1
        M = np.full((NT, NS), np.nan)
        M[g.teacher, g.student] = g.acc_port
        o = O[O.s == s]
        A = o[o.role == "teacher"].sort_values("idx").acc.to_numpy()
        S = o[o.role == "student"].sort_values("idx").acc.to_numpy()
        if kind != "insitu":
            A = np.full(NT, np.nan)
        out[s] = dict(port=M, A=A, S=S,
                      R_A=(M - FLOOR) / (A[:, None] - FLOOR),
                      R_B=(M - FLOOR) / (S[None, :] - FLOOR))
    return out


def two_way_boot_means(M, nboot, rng):
    NT, NS = M.shape
    ct = np.stack([np.bincount(r, minlength=NT) for r in rng.integers(0, NT, (nboot, NT))])
    cs = np.stack([np.bincount(r, minlength=NS) for r in rng.integers(0, NS, (nboot, NS))])
    return np.einsum("bi,ij,bj->b", ct, M, cs) / (NT * NS)


def mortality_bootstrap(mats, key="R_A", nboot=10000, seed=0, floor=True):
    rng = np.random.default_rng(seed)
    levels = sorted(mats)
    point = np.array([np.nanmean(mats[s][key]) for s in levels])
    boots = np.stack([two_way_boot_means(np.nan_to_num(mats[s][key], nan=0.0), nboot, rng)
                      for s in levels], 1)  # (nboot, L)
    p_fit = fit_hill(levels, point, floor)
    fits = np.array([fit_hill(levels, b, floor) for b in boots])
    sh_point = s_half(*p_fit)
    sh = np.array([s_half(*f) if np.all(np.isfinite(f)) else np.nan for f in fits])
    mu_b = 1 - boots
    rho = np.array([stats.spearmanr(levels, m).correlation for m in mu_b])
    return dict(levels=np.array(levels), point=point, boots=boots, fit=p_fit, fits=fits,
                s_half=sh_point, s_half_boot=sh, rho_point=stats.spearmanr(levels, 1 - point).correlation,
                rho_boot=rho, p_H1=float(np.mean(rho <= 0)))


def ci(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, q) if len(x) else np.array([np.nan, np.nan])


# ---------------------------------------------------------------- distillation
def distill_table(dist, mats, margin=0.02):
    rows = []
    for r in dist.itertuples():
        m = mats[r.s]
        port = m["port"][r.teacher, r.student]
        scratch = m["S"][r.student]
        denom = scratch - port
        D = (r.acc - port) / denom if denom > margin else np.nan
        rows.append(dict(r._asdict(), acc_port=port, acc_scratch=scratch, D=D))
    return pd.DataFrame(rows).drop(columns=["Index"])


def student_boot(df, value, stat, nboot=10000, seed=0):
    """Cluster bootstrap over students within each level; stat(df)->float."""
    rng = np.random.default_rng(seed)
    groups = {(s, j): g for (s, j), g in df.groupby(["s", "student"])}
    by_level = {}
    for (s, j) in groups:
        by_level.setdefault(s, []).append(j)
    out = []
    for _ in range(nboot):
        parts = []
        for s, js in by_level.items():
            pick = rng.choice(js, len(js), replace=True)
            parts += [groups[(s, j)] for j in pick]
        out.append(stat(pd.concat(parts)))
    return np.array(out)


def slope_logn(df, col="D"):
    g = df.groupby("n")[col].mean()
    return np.polyfit(np.log10(g.index.to_numpy(float)), g.to_numpy(), 1)[0]


def paired_diff(df, a, b, col="acc", ns=None):
    d = df if ns is None else df[df.n.isin(ns)]
    piv = d.pivot_table(index=["s", "student", "teacher", "n"], columns="cond", values=col)
    return float((piv[a] - piv[b]).mean())


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- Table 1
def calibration_table(sig, dyn, hp):
    rows = [
        ("sigma_g", sig.g, "lognormal coupling-gain spread at s=1",
         "derived: subthreshold current mismatch sigma_VT/(n U_T), A_VT~4 mV*um, W*L=1 um^2 "
         "(Pelgrom 1989; Kinget 2005) [verify for target process]"),
        ("sigma_o", sig.o, "coupling offset (units of w_max)", "assumed"),
        ("sigma_a, sigma_k", sig.a, "node gain / steepness spread",
         "derived: same mismatch mechanism as sigma_g; s=2 ~ 20% CV used for mixed-signal "
         "neuromorphic chips (Buchel 2021; Zendrikov 2023) [verify]"),
        ("sigma_c", sig.c, "node input offset (tanh linear-range units)",
         "derived: sigma_VOS ~ 4 mV over ~72 mV diff-pair linear range"),
        ("sigma_T", sig.T, "per-node effective temperature spread",
         "derived: shot-noise power scales with bias current => current mismatch"),
        ("sigma_r", dyn.sigma_r, "relative read noise (fixed across s)", "assumed"),
        ("sigma_prog", dyn.sigma_prog, "relative programming noise (fixed)", "assumed"),
        ("sigma_w", dyn.sigma_w, "absolute write noise, units of w_max (fixed)", "assumed"),
        ("alpha", dyn.alpha, "up/down update asymmetry (fixed)", "assumed"),
        ("w_max", f"{hp.wmax1}/{hp.wmax2}/{hp.wmaxb}", "weight bound W1/W2/bias",
         "design choice (tuned on nominal device)"),
        ("L", dyn.levels or "analog", "conductance levels", "analog in main; 64 in ablation"),
        ("nu_bar, sigma_nu", f"{dyn.nu_bar}, {dyn.sigma_nu}", "drift exponent (PCM-like)",
         "assumed; order of magnitude of PCM drift reports [verify]"),
        ("T", hp.T, "nominal temperature (energy units)", "tuned on nominal device"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "baseline", "meaning", "source"])
