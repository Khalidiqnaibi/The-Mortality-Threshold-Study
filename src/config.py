"""Hyperparameters and device-model baselines.

Everything here is static (hashable) so it can be passed to jit as a static arg.
Physical baselines (Sig, Dyn) are documented with their source in Table 1 of the
paper (analysis.calibration_table) and in prereg.md.
"""
from typing import NamedTuple
import yaml


class Sig(NamedTuple):
    """Static, device-specific mismatch spreads at s = 1 (the calibrated baseline).

    s = 1 corresponds to a subthreshold CMOS realisation with W*L = 1 um^2 and
    Pelgrom coefficient A_VT ~ 4 mV*um; s scales as 1/sqrt(W*L) (Pelgrom 1989).
    """
    g: float = 0.10     # lognormal coupling-gain spread (current-mirror mismatch)
    o: float = 0.01     # coupling offset / leakage, units of w_max (assumed)
    a: float = 0.10     # node output-gain spread (lognormal)
    b: float = 0.10     # node steepness spread (lognormal)
    c: float = 0.05     # node input offset, units of the tanh linear range
    T: float = 0.10     # per-node effective temperature spread (lognormal)


class Dyn(NamedTuple):
    """Dynamic, non-device-specific non-idealities (held fixed across s)."""
    sigma_r: float = 0.01      # relative read noise, drawn per relaxation run
    sigma_prog: float = 0.02   # relative programming noise
    sigma_w: float = 0.0005    # absolute write noise, units of w_max
    alpha: float = 0.10        # up/down update asymmetry (soft bounds)
    levels: int = 0            # 0 = analog; else number of conductance levels
    nu_bar: float = 0.05       # drift exponent mean (PCM-like), drift extension only
    sigma_nu: float = 0.01     # drift exponent spread


class HP(NamedTuple):
    H: int = 256
    T: float = 0.01            # nominal temperature
    dt: float = 0.25
    beta: float = 0.5
    n_burn_free: int = 20
    n_samp_free: int = 8
    n_burn_nudge: int = 8
    n_samp_nudge: int = 8
    lr1: float = 0.05
    lr2: float = 0.02
    wmax1: float = 0.25
    wmax2: float = 1.0
    wmaxb: float = 1.0
    batch: int = 32
    oracle: bool = False       # learning rule sees the coupling gains g_ij
    dyn_scale_with_s: bool = False  # ablation: s also scales dynamic noise
    n_eval_burn: int = 30
    n_eval_samp: int = 16

    def langevin_steps_per_update(self) -> int:
        return (self.n_burn_free + self.n_samp_free
                + 2 * (self.n_burn_nudge + self.n_samp_nudge))


def load_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    hp = HP(**d.get("hp", {}))
    sig = Sig(**d.get("sig", {}))
    dyn = Dyn(**d.get("dyn", {}))
    return hp, sig, dyn, d
