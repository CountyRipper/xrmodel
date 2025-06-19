import argparse
from llm import ModelConfig,LLMTrainer,KeyphrasePredictor,DataProcessor
from xmcdata import load_texts, load_sparse_matrix, load_label_text_map, csr_id_to_text
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XMC llm Model")
    parser.add_argument('--dataset_name', type=str, default='eurlex-4k', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-3B-Instruct', help='Name of the model')
    parser.add_argument('--train_type', type=str, default='train', help='Type of training to perform')
    parser.add_argument('--is_gen_labels',type=bool, default=False, help='Whether to generate labels or not')
    parser.add_argument('--reprocess_dataset', type=bool, default=False, help='Whether to reprocess the dataset or not')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    #model_name = "unsloth/Llama-3.2-3B-Instruct"
    #dataset_name = 'eurlex-4k'
    data_dir = f"xmc-base/{dataset_name}"
    label_map = load_label_text_map(data_dir + "/output-items.txt")
    # training dataset
    X_trn_text = load_texts(data_dir+"/X.trn.txt")
    Y_trn_feat = load_sparse_matrix(data_dir+"/Y.trn.npz")

    Y_trn_text,Y_trn_num = csr_id_to_text(Y_trn_feat,label_map)
    Y_trn_list= [",".join(y) for y in Y_trn_text]

    # validation dataset
    X_tst_text = load_texts(data_dir+"/X.tst.txt")
    Y_tst_feat = load_sparse_matrix(data_dir+"/Y.tst.npz")

    Y_tst_text, Y_tst_num = csr_id_to_text(Y_tst_feat,label_map)
    Y_tst_list = [",".join(y) for y in Y_tst_text]

    args.is_gen_labels = True
    stemmed_input_template = "Summarize the following document with keyphrases:\n\nDocument: {document}"
    normal_input_template = "Summarize the following document with keyphrases:\n\nDocument: {document}"
    stemmed_output_template = "Summary of this paragraph by unstemmed keyphrases: {keyphrases}"  # 输出模板
    output_template = "Summary of this paragraph by keyphrases: {keyphrases}"  # 输出模板
    llm_train_config = ModelConfig(
    model_name=args.model_name,  # 可以替换为其他模型如 "meta-llama/Llama-2-7b-hf"
    max_length=512,
    batch_size=8,
    learning_rate=2e-4,
    num_epochs=3,
    use_quantization=True,
    quantization_type="fp16",  # 可选: "int4", "int8", "fp16", "fp32"
    output_dir=f"./output/{dataset_name}/{args.model_name}",
    lora_r= 16,
    lora_alpha= 32,
    lora_dropout= 0.1,
    prompt_template=stemmed_input_template,
    max_new_tokens = 128 # 生成的最大新令牌数
    )
    trainer = LLMTrainer(llm_train_config)
    #加载模型
    # setting lora 
    trainer.setup_lora()
    # prepare dataset
    data_processor = DataProcessor(tokenizer=trainer.tokenizer,max_length_input=384,
                                max_length_output=128,  # 输出的最大长度
                                max_length = trainer.config.max_length,  # 输入的最大长度
                                prompt_template = stemmed_input_template,
                                res_template = stemmed_output_template
                               )
    print("args.reprocess_dataset",args.reprocess_dataset)
    if args.reprocess_dataset=='True':
        # 重新处理数据集
        train_dataset = data_processor.prepare_dataset(documents=X_trn_text,
                                               keyphrases=Y_trn_list,num_proc=8)
        val_dataset = data_processor.prepare_dataset(documents=X_tst_text,
                                             keyphrases=Y_tst_list,num_proc=8)
        data_processor.save_dataset(train_dataset, data_dir+"/train_dataset")
        data_processor.save_dataset(val_dataset, data_dir+"/val_dataset")
    else:
    # 从磁盘加载数据集
        train_dataset = data_processor.load_dataset(data_dir+"/train_dataset")
        val_dataset = data_processor.load_dataset(data_dir+"/val_dataset")
    if args.train_type == 'train':
        print("Training the model...")
        trainer.train(train_dataset,val_dataset)
        # 保存模型
        trainer.save_model(save_path=llm_train_config.output_dir)
    else:
        print(f"Loading the model for inference in {llm_train_config.output_dir}")
        trainer.load_trained_model(llm_train_config.output_dir)
    if args.is_gen_labels:
        # 生成标签
        predictor = KeyphrasePredictor(trainer = trainer)
        # 生成训练集标签
        results = predictor.batch_predict(X_tst_text,max_new_tokens=128,batch_size = 16,data_dir=data_dir+"/predicted_labels.txt")
        print("Labels generated and saved successfully.")
    else:
        print("Labels generation skipped.")
        print("Training completed and model saved successfully.")
        print(f"Model saved to {llm_train_config.output_dir}")
