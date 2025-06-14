from model import Seq2SeqModel
from xmcdata import load_texts, load_sparse_matrix, load_label_text_map, csr_id_to_text
from pathlib import Path  
from transformers import PreTrainedTokenizerBase
def seq2seq_predict(model: Seq2SeqModel, tokenizer:PreTrainedTokenizerBase,data_dir: str, output_dir: str = None):
    data_dir = Path(data_dir) # type: ignore

    X_tst_text = load_texts(str(data_dir / "X.tst.txt")) # type: ignore
    
    predictions = model.predict(
        tokenizer=tokenizer,
        inputs=X_tst_text,
        max_length=128,  # Adjust as needed
        num_beams=4,  # Adjust beam search parameters as needed
        output_dir=output_dir  # Optional output directory for saving predictions
    )

    # Convert predictions to text format
    pred_texts = [",".join(pred) for pred in predictions]

    return pred_texts