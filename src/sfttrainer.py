from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT for LLM")
    parser.add_argument('--dataset_name', type=str, default='eurlex-4k', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='Qwen1.5-0.5B', help='Name of the model')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    data_dir = f"xmc-base/{dataset_name}/"

    # 1. 加载数据
    dataset = load_dataset("csv", data_files={"train": data_dir+"train.csv", "test": data_dir+"test.csv"})

    # 2. 标签字符串 → 标签列表
    def preprocess(example):
        example["labels"] = example["labels"].split(",")
        return example

    dataset = dataset.map(preprocess)
    instruction_template = "Summarize this paragraph by keyphrases: {document}\n"
    # 3. 构造 prompt → response 格式
    def format_prompt(example):
        prompt = instruction_template.replace("{document}", example["document"])+"Answer:"
        #f"Summarize this paragraph by keyphrases: {example['document']}\n"
        response = ", ".join(example["labels"])
        return {"prompt": prompt, "completion": response}

    dataset = dataset.map(format_prompt)

    # 4. 加载模型和分词器
    model_name = "Qwen1.5-0.5B"  # 或任意支持 causal LM 的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name,   trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch. float16, device_map="auto", trust_remote_code=True)

    # 5. 标准化字段名
    #dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset = dataset.rename_columns({"prompt": "text", "completion":   "completion"})

    # 6. 数据 collator（只计算 completion 的 loss）
    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,     instruction_template=instruction_template, response_template="Answer:")

    # 7. 训练参数
    args = TrainingArguments(
        output_dir=data_dir+"sft_output/"+model_name,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=10,
        save_total_limit=2,
        eval_steps=300,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True,  # 如果你用 A100 or 4090 等
        remove_unused_columns=False,
    )

    # 8. 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=args,
        data_collator=collator,
        max_seq_length=512,
        #formatting_func=lambda x: f"Question: {x['text']}\nAnswer: {x   ['completion']}",
        formatting_func = lambda x: f"{x['text']}{x['completion']}"
    )

    # 9. 启动训练
    trainer.train()
    # 10. 保存模型
    trainer.save_model(data_dir+"sft_output/"+model_name)
    tokenizer.save_pretrained(data_dir+"sft_output/"+model_name)
    print(f"Model saved to {data_dir}sft_output/{model_name}")
    
