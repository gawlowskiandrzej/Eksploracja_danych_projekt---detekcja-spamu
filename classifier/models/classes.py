from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    csv_path: Path
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.2
    seed: int = 42

@dataclass
class ModelConfig:
    local_dir: Path = Path("models/llama3")
    device: str = "cuda"
    load_in_4bit: bool = True

@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    output_dir: str = "checkpoints"

@dataclass
class ClassifyConfig:
    threshold: float = 0.5
    temperature: float = 0.1
    max_new_tokens: int = 32