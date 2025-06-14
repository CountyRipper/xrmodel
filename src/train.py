from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from dataset import load_xmc_seq2seq_dataset
from model import Seq2SeqModel
from xmcdata import load_texts, load_sparse_matrix, load_label_text_map, csr_id_to_text
from pathlib import Path 
from model import seq2seqParams, Seq2SeqModel

if __name__ == "__main__":
    dataset_name = 'eurlex-4k'
    model_name = 't5'
    params_path = f'./params/{dataset_name}/{model_name}.json'
    data_dir = "xmc-base/eurlex-4k"
    label_map = load_label_text_map(data_dir + "/output-items.txt")
    prefix = 'Summary this paragraph: '

    X_trn_text = load_texts(data_dir+"/X.trn.txt")
    X_trn_text = [prefix + text for text in     X_trn_text]
    Y_trn_text,Y_trn_num = csr_id_to_text(load_sparse_matrix(data_dir+"/Y.trn.npz"),label_map)
    Y_trn_text = [",".join(y) for y in Y_trn_text]

    X_tst_text = load_texts(data_dir+"/X.tst.txt")
    X_tst_text = [prefix + text for text in X_tst_text]
    Y_tst_text, Y_tst_num = csr_id_to_text(load_sparse_matrix(data_dir+"/Y.tst.npz"),label_map)
    Y_tst_text = [",".join(y) for y in Y_tst_text]

    args = seq2seqParams.load_config(params_path)
    args.batch_size = 10
    model =  Seq2SeqModel(args)

    train_dataset, val_dataset = load_xmc_seq2seq_dataset(
        X_trn_text, Y_trn_text, X_tst_text, Y_tst_text, model.tokenizer,max_length=args.max_input_length
    )
    output_dir = f"output/{dataset_name}/{model_name}"
    trainer, _ = model.gen_train(train_dataset, val_dataset,output_dir)
    model.model.from_pretrained(output_dir)

