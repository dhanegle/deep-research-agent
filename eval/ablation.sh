#!/bin/bash
# 消融实验矩阵：4 组配置 × 2 任务 × 3 次重复，local 搜索源（零 API 消耗）
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python

for i in 1 2 3; do
  echo "===== round $i ====="
  $PY eval/run_eval.py --provider local --limit 2 --no-faith --tag "full-r$i"
  $PY eval/run_eval.py --provider local --limit 2 --no-faith --no-reflect --tag "no-reflect-r$i"
  $PY eval/run_eval.py --provider local --limit 2 --no-faith --react-mode prompt --tag "prompt-react-r$i"
  $PY eval/run_eval.py --provider local --limit 2 --no-faith --naive --tag "naive-r$i"
done
echo "ALL DONE"
