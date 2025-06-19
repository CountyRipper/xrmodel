nohup python ./src/run_llm.py \
  --dataset_name "eurlex-4k" \
  --model_name "unsloth/Llama-3.2-3B-Instruct" \
  --train_type "eval" \
  --is_gen_labels True \
  --reprocess_dataset False \
  > log.txt 2>&1 &