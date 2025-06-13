from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from params import Params
from dataset import load_xmc_seq2seq_dataset
from model import Seq2SeqModel
from xmcdata import load_texts, load_sparse_matrix, load_label_text_map, csr_id_to_text
from pathlib import Path 
def train(config: Params):
    data_dir = config.data_dir
    data_dir = Path(data_dir)
    label_map = load_label_text_map(str(data_dir/"output-items.txt"))
    X_trn_text = load_texts(str(data_dir/"X.trn.txt"))
    Y_trn_text, _ = csr_id_to_text(load_sparse_matrix(str(data_dir/"Y.trn.npz")), label_map)
    X_val_text = load_texts(str(data_dir/"X.tst.txt"))
    Y_val_text, _ = csr_id_to_text(load_sparse_matrix(str(data_dir/"Y.tst.npz")), label_map)
    Y_trn_text = [",".join(y) for y in Y_trn_text]
    Y_val_text = [",".join(y) for y in Y_val_text]

    train_dataset, val_dataset, tokenizer = load_xmc_seq2seq_dataset(X_trn_text, Y_trn_text, X_val_text, Y_val_text, config.model_name_or_path, max_length=config.max_input_length)
    model = Seq2SeqModel(config.model_name_or_path)  # unwrap for Trainer

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        save_strategy="epoch",
        fp16=config.use_fp16,
        logging_steps=config.logging_steps,
        logging_dir=f"./{config.output_dir}/logs",
        report_to=["tensorboard"],
        save_safetensors=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model for XMC")
    parser.add_argument("--params_path", type=str, required=True, help="Path to the configuration JSON file")
    args = parser.parse_args()

    config = Params.from_json(args.params_path)
    train(config)

