#!/bin/bash
# Full pipeline: main study (MNIST) then Fashion-MNIST replication, then report.
set -e
cd "$(dirname "$0")"
PY=../.venv/Scripts/python
$PY run_study.py configs/main.yaml train port readouts distill
$PY run_study.py configs/fashion.yaml train port
$PY make_report.py configs/main.yaml configs/fashion.yaml
$PY run_study.py configs/main.yaml ablate
$PY make_report.py configs/main.yaml configs/fashion.yaml
echo ALL_DONE
