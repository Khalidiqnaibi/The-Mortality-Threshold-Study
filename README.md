# The Mortality Threshold Study

Simulation code and results for *The Mortality Threshold: Weight Portability and Output-Only Distillation Between Mismatched Analog Devices Trained by Equilibrium Propagation*.

A layered Hopfield-type network is trained in situ by symmetric equilibrium propagation on simulated analog devices with static mismatch. The study measures how well trained weights carry over to another device as mismatch grows, and how output-only distillation compares with copying weights and with a robust train-once baseline.

## Layout

- `src/` simulator (energy and Langevin dynamics, device model, equilibrium propagation, training, analysis)
- `tests/` validation tests V1-V8 and a leakage test
- `configs/` experiment configurations (`main.yaml`, `fashion.yaml`)
- `run_study.py`, `run_sensitivity.py`, `run_sens_distill.py` experiment drivers
- `make_report.py` regenerates tables and figures from `results/`
- `results/` result tables (parquet) and logs. Trained weight files are not included.

## Running

Python 3.12 with JAX (CPU), NumPy, SciPy, pandas, pyarrow, matplotlib, PyYAML, tabulate.

```bash
python -m pytest -q tests
bash run_all.sh
```

MNIST goes in `data/mnist.npz`; Fashion-MNIST idx files go in `data/` as `fashion-*.gz`. The full run takes about 10 hours on a six-core CPU.
