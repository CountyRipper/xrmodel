import json
from dataclasses import dataclass, field, asdict

@dataclass
class Params:
    # 模型与路径相关
    model_name_or_path: str
    data_dir: str # dataset_dir name 
    output_dir: str # output_dir name

    # 超参数
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_input_length: int = 512
    max_target_length: int = 50

    # 其他
    use_fp16: bool = True
    logging_steps: int = 10

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
