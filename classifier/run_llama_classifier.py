import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from classifier.implementations.classifier import LlamaSpamClassifier
from classifier.implementations.prompt_builder import LlamaSpamPromptBuilder
from classifier.implementations.csv_data_loader import CsvDataLoader
from classifier.implementations.llama_model import LlamaModel
from classifier.implementations.qlora_trainer import QLoRATrainer
from classifier.models.classes import TrainingConfig, DataConfig, ModelConfig
from config import main_dataset, llama_cfg, classify_cfg, llama_train_cfg

#python -X utf8 .\classifier\run_llama_classifier.py 
import os
os.environ["PYTHONUTF8"] = "1"

def setup_model(llama_cfg: ModelConfig) -> LlamaModel:
    llama          = LlamaModel()
    llama.load(llama_cfg)
    return llama
def train(llama: LlamaModel, prompt_builder: LlamaSpamPromptBuilder, df_train: pd.DataFrame, df_test: pd.DataFrame, train_cfg: TrainingConfig, data_cfg: DataConfig):
    trainer = QLoRATrainer(llama, prompt_builder)
    trainer.train(df_train, df_test, train_cfg, data_cfg)
    trainer.save(f"{train_cfg.output_dir}/final")
    return trainer.evaluate()

def main() -> None:
    loader = CsvDataLoader()
    prompt_builder = LlamaSpamPromptBuilder()
    raw_data = loader.load(main_dataset)
    data = loader.preprocess(raw_data, main_dataset)
    df_train, df_test = loader.split(data, main_dataset)
    llama = setup_model(llama_cfg)
    adapter_path = Path(f"{llama_train_cfg.output_dir}/final")
    
    if adapter_path.exists():
        print(f"Adapter już istnieje pod ścieżką: {adapter_path}. Pomijam trening.")
    else:
        print(f"Adapter nie istnieje. Uruchamiam trening i zapiszę do: {adapter_path}.")
        train(llama, prompt_builder, df_train, df_test, llama_train_cfg, main_dataset)
    
    classifier = LlamaSpamClassifier(llama, prompt_builder, classify_cfg)
    classifier.fit([], [], adapter_path=str(adapter_path))
    x_test, y_test = loader.to_lists(df_test, main_dataset)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
if __name__ == "__main__":
    main()
