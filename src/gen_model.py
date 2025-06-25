import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    T5TokenizerFast,
    BartForConditionalGeneration,
    BartTokenizer,
    BartTokenizerFast,
    PegasusTokenizerFast,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import warnings

@dataclass
class T2TConfig:
    """Universal configuration class for Text-to-Text models"""
    # Model configuration
    model_name: str = "google/flan-t5-base"  # Can be T5, BART, or Pegasus models
    model_type: str = "auto"  # auto, t5, bart, pegasus
    
    # Data configuration
    dataset_dir: str = "./xmc-base/"
    output_dir: str = "./outputs"
    # Prompt configuration - T5 style strict formatting
    prompt: str = "Please analyze the following document and provide the appropriate label:"
    task_prefix: str = ""  # For T5: "summarize:", "translate:", etc.
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    warmup_ratio: float = 0.1
    max_input_length: int = 512
    max_output_length: int = 128
    
    # Tokenization standards (T5-style strict rules)
    strict_tokenization: bool = True  # Use T5-style strict tokenization standards
    normalize_text: bool = True  # Normalize text following T5 standards
    add_special_tokens: bool = True
    
    # Generation configuration
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    length_penalty: float = 1.0
    
    # Other configuration
    save_strategy: str = "epoch"  # 'epoch' or 'steps'
    evaluation_strategy: str = "epoch"  # 'no', 'epoch', 'steps
    save_total_limit: Optional[int] = 1
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42
    bf16: bool = False
    use_tensorboard: bool = True

class UniversalTokenizer:
    """Universal tokenizer wrapper that applies T5-style strict standards to all models"""
    
    def __init__(self, model_name: str, config: T2TConfig):
        self.config = config
        self.model_type = self._detect_model_type(model_name)
        
        # Load appropriate tokenizer
        if self.model_type == "t5":
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        elif self.model_type == "bart":
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        elif self.model_type == "pegasus":
            self.tokenizer = PegasusTokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"Loaded {self.model_type.upper()} tokenizer: {model_name}")
        
    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from model name"""
        if self.config.model_type != "auto":
            return self.config.model_type.lower()
        
        model_name_lower = model_name.lower()
        if "t5" in model_name_lower or "flan" in model_name_lower:
            return "t5"
        elif "bart" in model_name_lower:
            return "bart"
        elif "pegasus" in model_name_lower:
            return "pegasus"
        else:
            warnings.warn(f"Could not auto-detect model type for {model_name}, defaulting to T5")
            return "t5"
    
    def _normalize_text_t5_style(self, text: str) -> str:
        """Apply T5-style text normalization (strict standards)"""
        if not self.config.normalize_text:
            return text
        
        # T5-style normalization rules
        text = text.strip()
        # Remove extra whitespaces (T5 standard)
        text = ' '.join(text.split())
        # Ensure proper sentence endings for T5
        if text and not text.endswith(('.', '!', '?', ':')):
            text += '.'
        
        return text
    
    def _format_input_t5_style(self, text: str) -> str:
        """Format input text following T5 conventions"""
        # Normalize text
        text = self._normalize_text_t5_style(text)
        
        # Add task prefix if specified (T5 style)
        if self.config.task_prefix:
            text = f"{self.config.task_prefix} {text}"
        
        # Add prompt (following T5 format standards)
        if self.config.prompt:
            text = f"{self.config.prompt} {text}"
        
        return text
    
    def _format_target_t5_style(self, text: str) -> str:
        """Format target text following T5 conventions"""
        return self._normalize_text_t5_style(text)
    
    def encode_input(self, text: str, **kwargs) -> Dict:
        """Encode input text with T5-style strict standards"""
        # Apply T5-style formatting
        formatted_text = self._format_input_t5_style(text)
        
        # Tokenize with strict T5 standards
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=self.config.add_special_tokens,
            max_length=kwargs.get('max_length', self.config.max_input_length),
            padding=kwargs.get('padding', 'max_length'),
            truncation=kwargs.get('truncation', True),
            return_tensors=kwargs.get('return_tensors', 'pt')
        )
        
        return encoding
    
    def encode_target(self, text: str, **kwargs) -> Dict:
        """Encode target text with T5-style strict standards"""
        # Apply T5-style formatting
        formatted_text = self._format_target_t5_style(text)
        
        # For T5, use as_target_tokenizer context
        if self.model_type == "t5":
            with self.tokenizer.as_target_tokenizer():
                encoding = self.tokenizer(
                    formatted_text,
                    add_special_tokens=self.config.add_special_tokens,
                    max_length=kwargs.get('max_length', self.config.max_output_length),
                    padding=kwargs.get('padding', 'max_length'),
                    truncation=kwargs.get('truncation', True),
                    return_tensors=kwargs.get('return_tensors', 'pt')
                )
        else:
            # For BART/Pegasus, regular tokenization
            encoding = self.tokenizer(
                formatted_text,
                add_special_tokens=self.config.add_special_tokens,
                max_length=kwargs.get('max_length', self.config.max_output_length),
                padding=kwargs.get('padding', 'max_length'),
                truncation=kwargs.get('truncation', True),
                return_tensors=kwargs.get('return_tensors', 'pt')
            )
        
        return encoding
    
    def batch_decode(self, token_ids, **kwargs):
        """Decode tokens with consistent output formatting"""
        return self.tokenizer.batch_decode(token_ids, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        """Decode single sequence"""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def save_pretrained(self, path):
        return self.tokenizer.save_pretrained(path)

class CustomDataset(Dataset):
    """Universal dataset class for CSV files with strict T5-style processing"""
    
    def __init__(self, csv_file: str, tokenizer: UniversalTokenizer, config: T2TConfig, is_training: bool = True, cache_file: str = None):
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        if cache_file and os.path.exists(cache_file):
            print(f"Loading tokenized data from cache:{cache_file}")
            self.load_cache(cache_file)
            return
        # Load CSV data with strict validation
        print(f"Loading data from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        
        # Strict validation following T5 standards
        required_cols = ["document", "labels"]
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}. Found: {list(self.data.columns)}")
        
        # Apply strict T5-style data cleaning
        self.data = self._clean_data_t5_style(self.data)
        
        print(f"Loaded {len(self.data)} samples after T5-style cleaning")
        
        # Tokenize data with progress bar using strict standards
        self._tokenize_data_strict()
        # if has cache_file
        if cache_file:
            self.save_cache(cache_file)
    
    def save_cache(self, file_path: str):
        """把 tokenize 后的 tensors 列表保存到磁盘"""
        torch.save({
            'input_ids': self.input_ids,
            'attention_masks': self.attention_masks,
            'labels':       self.labels,
        }, file_path)
        print(f"Tokenized data saved to {file_path}")

    def load_cache(self, file_path: str):
        """从磁盘读取 tokenized 数据，跳过 CSV/clean/tokenize"""
        data = torch.load(file_path)
        self.input_ids     = data['input_ids']
        self.attention_masks = data['attention_masks']
        self.labels        = data['labels']
        print(f"Loaded {len(self.input_ids)} samples from cache")
    
    def _clean_data_t5_style(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply T5-style strict data cleaning"""
        print("Applying T5-style strict data cleaning...")
        
        original_len = len(data)
        
        # Convert to string and handle NaN values (T5 standard)
        data["document"] = data["document"].astype(str).fillna("")
        data["labels"] = data["labels"].astype(str).fillna("")
        
        # Remove empty documents (T5 strict standard)
        data = data[data["document"].str.strip() != ""]
        data = data[data["labels"].str.strip() != ""]
        
        # Remove documents that are too short (T5 quality standard)
        data = data[data["document"].str.len() >= 10]
        data = data[data["labels"].str.len() >= 1]
        
        # Reset index
        data = data.reset_index(drop=True)
        
        print(f"Data cleaning: {original_len} -> {len(data)} samples")
        return data
    
    def _tokenize_data_strict(self):
        """Tokenize all data with T5-style strict standards and progress bar"""
        print("Tokenizing data with T5-style strict standards...")
        
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for idx in tqdm(range(len(self.data)), desc="Strict Tokenization"):
            document = self.data.iloc[idx]['document']
            label = self.data.iloc[idx]['labels']
            
            # Encode input with strict T5 standards
            input_encoding = self.tokenizer.encode_input(document)
            
            # Encode target with strict T5 standards
            target_encoding = self.tokenizer.encode_target(label)
            
            self.input_ids.append(input_encoding.input_ids.squeeze())
            self.attention_masks.append(input_encoding.attention_mask.squeeze())
            
            # Handle labels based on model type (T5 strict standard)
            labels = target_encoding.input_ids.squeeze()
            if self.tokenizer.model_type == "t5":
                # T5 uses -100 for padding tokens
                labels[labels == self.tokenizer.pad_token_id] = -100
            
            self.labels.append(labels)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx]
        }

class UniversalT2TModel:
    """Universal Text-to-Text model wrapper supporting T5, BART, and Pegasus"""
    
    def __init__(self, config: T2TConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize universal tokenizer
        self.tokenizer = UniversalTokenizer(config.model_name, config)
        
        # Load appropriate model
        self.model = self._load_model(config.model_name, self.tokenizer.model_type)
        self.model.to(self.device)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Model loaded on {self.device}")
        print(f"Model type: {self.tokenizer.model_type.upper()}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_name: str, model_type: str):
        """Load appropriate model based on type"""
        print(f"Loading {model_type.upper()} model: {model_name}")
        
        if model_type == "t5":
            return T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_type == "bart":
            return BartForConditionalGeneration.from_pretrained(model_name)
        elif model_type == "pegasus":
            return PegasusForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    def load_local_model(self, model_path: str):
        """Load a local model from the specified path"""
        print(f"Loading local model from {model_path}...")

        if self.tokenizer.model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif self.tokenizer.model_type == "bart":
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
        elif self.tokenizer.model_type == "pegasus":
            self.model = PegasusForConditionalGeneration.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.tokenizer.model_type}")

        self.model.to(self.device)
        print(f"Local model loaded successfully from {model_path}")
    
    def load_datasets(self):
        """Load train and test datasets with strict T5-style processing"""
        train_file = Path(self.config.dataset_dir) / "train.csv"
        test_file = Path(self.config.dataset_dir) / "test.csv"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        self.train_dataset = CustomDataset(
            str(train_file), 
            self.tokenizer, 
            self.config, 
            cache_file= self.config.dataset_dir+"/train.pt",
            is_training=True
        )
        
        self.test_dataset = CustomDataset(
            str(test_file), 
            self.tokenizer, 
            self.config, 
            cache_file= self.config.dataset_dir+"/test.pt",
            is_training=False
        )
        
        return self.train_dataset, self.test_dataset
    
    def train(self):
        """Train the model with T5-style strict standards"""
        print("Starting training with T5-style strict standards...")
        
        # Load datasets
        train_dataset, val_dataset = self.load_datasets()
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            #warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            logging_dir=self.config.output_dir,
            #eval_steps=self.config.eval_steps,
            save_total_limit= self.config.save_total_limit,
            save_strategy=self.config.save_strategy,
            eval_strategy= self.config.evaluation_strategy,
            remove_unused_columns=False,
            push_to_hub=False,
            bf16= self.config.bf16,
            report_to=["tensorboard"] if self.config.use_tensorboard else []
        )
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset= val_dataset,
            processing_class=self.tokenizer.tokenizer,  # Use underlying tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Training completed. Model saved to {self.config.output_dir}")
        return self.model
    
    def generate_predictions(self, 
                           inputs: List[str], 
                           batch_size: int = None,
                           output_path: str = None,
                           max_output_length: int = None,
                           num_beams: int = None,
                           is_train: bool = False) -> List[str]:
        """
        Generate predictions with T5-style strict processing and real-time file writing
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        if max_output_length is None:
            max_output_length = self.config.max_output_length
        if num_beams is None:
            num_beams = self.config.num_beams
        
        self.model.eval()
        results = []
        
        # Prepare output file
        if output_path:
            output_file = Path(output_path) / f"test_gen_pred_{str(self.config.model_name).split('/')[-1]}.txt"
            if is_train:
                output_file = Path(output_path) / f"train_gen_pred_{str(self.config.model_name).split('/')[-1]}.txt"
            # Create directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Clear file
            with open(str(output_file), 'w+') as f:
                f.write("")
        
        # Generate in batches with T5-style strict processing
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating with T5 Standards"):
            batch = inputs[i:i+batch_size]
            
            # Process batch through universal tokenizer (T5-style strict)
            batch_encodings = []
            for inp in batch:
                encoding = self.tokenizer.encode_input(
                    inp,
                    max_length=self.config.max_input_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt"
                )
                batch_encodings.append(encoding)
            
            # Pad batch manually for consistent processing
            max_len = max(enc.input_ids.shape[1] for enc in batch_encodings)
            input_ids = []
            attention_masks = []
            
            for enc in batch_encodings:
                # Pad to max length in batch
                pad_len = max_len - enc.input_ids.shape[1]
                if pad_len > 0:
                    input_ids.append(torch.cat([
                        enc.input_ids, 
                        torch.full((1, pad_len), self.tokenizer.pad_token_id)
                    ], dim=1))
                    attention_masks.append(torch.cat([
                        enc.attention_mask,
                        torch.zeros((1, pad_len))
                    ], dim=1))
                else:
                    input_ids.append(enc.input_ids)
                    attention_masks.append(enc.attention_mask)
            
            # Stack tensors
            batch_input_ids = torch.cat(input_ids, dim=0).to(self.device)
            batch_attention_masks = torch.cat(attention_masks, dim=0).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    max_length=max_output_length,
                    num_beams=num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    length_penalty=self.config.length_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode batch with T5-style post-processing
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Apply T5-style post-processing to outputs
            decoded = [self._post_process_output_t5_style(text) for text in decoded]
            
            results.extend(decoded)
            
            # Write to file immediately
            if output_path:
                with open(str(output_file), 'a', encoding='utf-8') as f:
                    for text in decoded:
                        f.write(text.strip(",") + "\n")
        
        return results
    
    def _post_process_output_t5_style(self, text: str) -> str:
        """Apply T5-style post-processing to generated text"""
        # T5-style output cleaning
        text = text.strip()
        # Remove any remaining special tokens
        text = text.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
        # Clean extra whitespace
        text = ' '.join(text.split())
        return text
    
    def predict_test_set(self, output_path: str = None):
        """Generate predictions for the test set with T5-style processing"""
        # Load test dataset
        _, test_dataset = self.load_datasets()
        
        # Extract documents from test dataset (with T5-style cleaning already applied)
        test_data = pd.read_csv(Path(self.config.dataset_dir) / "test.csv")
        documents = test_data["document"].astype(str).fillna("").tolist()
        
        # Generate predictions
        if output_path is None:
            output_path = self.config.output_dir
        
        predictions = self.generate_predictions(
            documents, 
            output_path=output_path
        )
        
        return predictions
    
    def predict_train_set(self, output_path: str = None):
        """Generate predictions for the test set with T5-style processing"""
        # Load test dataset
        _, test_dataset = self.load_datasets()
        
        # Extract documents from test dataset (with T5-style cleaning already applied)
        test_data = pd.read_csv(Path(self.config.dataset_dir) / "train.csv")
        documents = test_data["document"].astype(str).fillna("").tolist()
        
        # Generate predictions
        if output_path is None:
            output_path = self.config.output_dir
        
        predictions = self.generate_predictions(
            documents, 
            output_path=output_path,
            is_train=True
        )
        
        return predictions
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_path = Path(self.config.output_dir) / "config.json"
        config_dict = self.config.__dict__.copy()
        config_dict['detected_model_type'] = self.tokenizer.model_type
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {config_path}")

# Example usage for different models
def main():
    """Main function demonstrating usage with different T2T models"""
    
    # Example configurations for different models
    data_dir = "./xmc-base/"
    dataset_name = "wiki10-31k"
    dataset_dir = data_dir+dataset_name
    model_names = [
        "google/flan-t5-base",  # T5-base
        "google/flan-t5-large",  # T5-large
        "google/flan-t5-xl",     # T5-xl
        "facebook/bart-base",    # BART
        "google/pegasus-xsum"    # Pegasus
    ]
    # T5-FLAN Configuration
    t5_config = T2TConfig(
        model_name=model_names[0],  # Change to other   models as needed
        dataset_dir=dataset_dir,
        output_dir=dataset_dir+"/outputs/"+model_names[0].split("/")[-1],
        prompt="Summarize this document by unstemmed keyphrases:",
        #task_prefix="classify:",  # T5-style task  prefix
        num_epochs=5,
        batch_size=16, # base-16, large-4, xl-2? for    24GB
        learning_rate= 5e-5,
        max_input_length=512,
        max_output_length=128,
        strict_tokenization=True,
        warmup_ratio= 0.05,
        fp16=False, #t5 cannot use fp16
        bf16= True,
    )

    # BART Configuration
    bart_config = T2TConfig(
        model_name=model_names[3],  # Change to other   models as needed
        dataset_dir=dataset_dir,
        output_dir=dataset_dir+"/outputs/"+model_names[3].split("/")[-1],
        prompt="Summarize this document by unstemmed keyphrases:",
        num_epochs=3,
        batch_size=4,
        max_input_length=512,
        max_output_length=128,
        learning_rate= 2e-5,
        fp16=True,
        bf16= False,
        strict_tokenization=True  # Apply T5-style  strict standards to BART
    )

    # Pegasus Configuration
    pegasus_config = T2TConfig(
        model_name=model_names[4],  # Change to other   models as needed
        dataset_dir=dataset_dir,
        output_dir=dataset_dir+"/outputs/"+model_names[4].split("/")[-1],
        prompt="Summarize this document by unstemmed keyphrases:",
        num_epochs=3,
        batch_size=4,  # Smaller batch for Pegasus
        max_input_length=512,
        max_output_length=128,
        learning_rate= 2e-5,
        fp16=True,
        bf16= False,
        strict_tokenization=True  # Apply T5-style  strict standards to Pegasus
    )
    
    # Choose configuration
    config = t5_config  # Change to bart_config or pegasus_config as needed
    
    # Initialize model
    model = UniversalT2TModel(config)
    
    # Save configuration
    model.save_config()
    
    # Train the model (uncomment to train)
    model.train()
    
    # Generate predictions for test set
    print("Generating predictions with T5-style strict processing...")
    predictions = model.predict_test_set(output_path=dataset_dir)
    
    print(f"Generated {len(predictions)} predictions using {model.tokenizer.model_type.upper()}")
    print("Sample predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"{i+1}: {pred}")

if __name__ == "__main__":
    main()