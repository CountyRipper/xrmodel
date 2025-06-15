import argparse
from dataset import load_xmc_seq2seq_dataset, load_xmc_seq2seq_dataset_from_disk
from model import Seq2SeqModel
from xmcdata import load_texts, load_sparse_matrix, load_label_text_map, csr_id_to_text
from pathlib import Path 
from model import seq2seqParams, Seq2SeqModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XMC Seq2Seq Model")
    parser.add_argument('--dataset_name', type=str, default='wiki10', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='t5', help='Name of the model')
    parser.add_argument('--train_type', type=str, default='train', help='Type of training to perform')
    parser.add_argument('--is_gen_labels',type=bool, default=False, help='Whether to generate labels or not')
    parser.add_argument('--reprocess_dataset', type=bool, default=False, help='Whether to reprocess the dataset or not')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    if dataset_name not in ['eurlex-4k', 'wiki10-31k', 'amazon-13k', 'wiki-500k']:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    model_name = args.model_name
    if model_name not in ['t5', 'bart', 'pegasus',
                          't5-large','bart-large','pegasus-large',
                          'flan-t5', 'flan-t5-large',
                          'flan-t5-xl', 'flan-t5-xxl',
                          't5-base', 't5-small', 't5-3b',
                          'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        raise ValueError(f"Unsupported model: {model_name}")
        
    
    params_path = f'./params/{dataset_name}/{model_name}.json'
    data_dir = f"xmc-base/{dataset_name}"

    print("preparing data for", dataset_name, model_name)

    label_map = load_label_text_map(data_dir + "/output-items.txt")
    prefix = 'Summarization of this paragraph: '
    
    X_trn_text = load_texts(data_dir+"/X.trn.txt")
    #X_trn_text = [prefix + text for text in X_trn_text]
    Y_trn_feat = load_sparse_matrix(data_dir+"/Y.trn.npz")
    
    Y_trn_text,Y_trn_num = csr_id_to_text(load_sparse_matrix(data_dir+"/Y.trn.npz"),label_map)
    Y_trn_text = [",".join(y) for y in Y_trn_text]
    Y_trn_text = [prefix + text for text in Y_trn_text]
    
    X_tst_text = load_texts(data_dir+"/X.tst.txt")
    Y_tst_feat = load_sparse_matrix(data_dir+"/Y.tst.npz")
    #X_tst_text = [prefix + text for text in X_tst_text]
    Y_tst_text, Y_tst_num = csr_id_to_text(load_sparse_matrix(data_dir+"/Y.tst.npz"),label_map)
    Y_tst_text = [",".join(y) for y in Y_tst_text]
    Y_tst_text = [prefix + text for text in Y_tst_text]

    train_args = seq2seqParams.load_config(params_path)
    model =  Seq2SeqModel(train_args)
    print("run_args", args.__str__())
    print("train_args", train_args.__str__())
    
    if args.reprocess_dataset or( Path(data_dir).joinpath('val_dataset').exists() and Path(data_dir).joinpath('train_dataset').exists() ):
        print("have preprocessed data, loading from disk")
        train_dataset, val_dataset = load_xmc_seq2seq_dataset_from_disk(data_dir)
    else:
        print("processing data for seq2seq")
        train_dataset, val_dataset = load_xmc_seq2seq_dataset(
        X_trn_text, Y_trn_text, X_tst_text, Y_tst_text_val, model.tokenizer,max_length=train_args.max_input_length,save_dir=data_dir)

    output_dir = f"output/{dataset_name}/{model_name}"
    if args.train_type == 'train':
        print("training model")
        trainer, _ = model.gen_train(train_dataset, val_dataset,output_dir)
    else:
        print("preparing model")
        model.model.from_pretrained(output_dir)
    
    if args.is_gen_labels:
        print("generating labels")
        model.predict(X_tst_text,batch_size=8,max_input_length=train_args.max_input_length,num_beams=train_args.num_beams,output_path=Path(data_dir))

    print("done")
    print("output_dir", output_dir)
