#! /bin/bash
set -e 

source .venv/bin/activate
nohup python train.py > train.log 2>&1 &
echo "Train Task Submitted, check train.log for details"