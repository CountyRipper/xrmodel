#!bin/bash
nohup python ./src/run.py \
  --dataset_name "wiki10-31k" \
  --model_name "t5" \
  --train_type "train" \
  --is_gen_labels True \
  --reprocess_dataset True \
  > log.txt 2>&1 &