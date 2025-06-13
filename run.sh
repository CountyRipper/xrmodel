#!bin/bash
nohup python ./src/train.py --params_path "./params/eurlex-4k/bart.json" > log.txt 2>&1 &