import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoTokenizer

from classifier.implementations.classifier import LlamaSpamClassifier
from classifier.implementations.prompt_builder import LlamaSpamPromptBuilder
from classifier.implementations.csv_data_loader import CsvDataLoader
from classifier.implementations.llama_model import LlamaModel
from classifier.implementations.qlora_trainer import QLoRATrainer
from classifier.models.classes import ClassifyConfig,TrainingConfig, DataConfig, ModelConfig
from classifier.implementations.sklearn_classifier import SklearnClassifier


import os
os.environ["PYTHONUTF8"] = "1"

def build_classifiers() -> list[SklearnClassifier]:
    return [
        SklearnClassifier(
            "Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)
        ),
        SklearnClassifier(
            "Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)
        ),
        SklearnClassifier(
            "Support Vector Machine",
            SVC(kernel="linear", probability=True, random_state=42),
        ),
        SklearnClassifier("Naive Bayes", MultinomialNB()),
        SklearnClassifier(
            "Gradient Boosting", GradientBoostingClassifier(random_state=42)
        ),
        SklearnClassifier("Decision Tree", DecisionTreeClassifier(random_state=42)),
        SklearnClassifier("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ]


def evaluate_model(name: str, y_true: list, y_pred: list) -> dict:
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def setup_model(llama_cfg: ModelConfig) -> LlamaModel:
    llama          = LlamaModel()
    llama.load(llama_cfg)
    return llama
def train(llama: LlamaModel, prompt_builder: LlamaSpamPromptBuilder, df_train: pd.DataFrame, df_test: pd.DataFrame, train_cfg: TrainingConfig, data_cfg: DataConfig):
    trainer = QLoRATrainer(llama, prompt_builder)
    trainer.train(df_train, df_test, train_cfg, data_cfg)
    trainer.save(f"{train_cfg.output_dir}/final")
    return trainer.evaluate()

def print_results_table(all_results: dict):
    """
    all_results: { "dataset_name": [{"name": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...}] }
    """
    for dataset, results in all_results.items():
        print(f"\nDataset: {dataset}")
        df = pd.DataFrame(results).set_index("name")
        df.index.name = "Model"
        print(df[["accuracy", "precision", "recall", "f1"]].round(4).to_string())


def main() -> None:
    # --- TRAINING SETUP ---
    # 1. Podaj lokalną ścieżkę do modelu LLaMA w llama_cfg.local_dir.
    # 2. Podaj ścieżkę do datasetu w config1.csv_path.
    # 3. Określ proporcję zbioru testowego przez test_size (np. 0.2 = 20%).
    # 4. Po wytrenowaniu zakomentuj poniższy wywołanie train() i upewnij się,
    #    że adapter_path w classifier.fit() wskazuje na prawidłowy katalog z
    #    zapisanym adapterem/fintunowanym modelem.
    # 5. Próbka testowa do oceny jest określona przez zmienną sample.
    config1 = DataConfig(
            csv_path=Path(f"data/llm_spam_email_dataset.csv"),
            text_column="email_text",
            label_column="label",
            test_size=0.2,
            seed=42,
        )
    # config2 = DataConfig(
    #         csv_path=Path(f"data/email_dataset_100k.csv"),
    #         text_column="body_plain",
    #         label_column="label",
    #         test_size=0.2,
    #         seed=42,
    #     )
    # config3 = DataConfig(
    #         csv_path=Path(f"generated_dataset_v1.csv"),
    #         text_column="email_text",
    #         label_column="label",
    #         separator=";",
    #         test_size=0.2,
    #         seed=42,
    #     )
    all_results = {}
    # for config in [config1, config2, config3]:
    #     print(f"Evaluating classifiers on {config.csv_path.name}...")
    #     loader = CsvDataLoader()
    #     raw_data = loader.load(config)
    #     data = loader.preprocess(raw_data, config)
    #     df_train, df_test = loader.split(raw_data, config)
    #     x_test, y_test = loader.to_lists(df_test, config)
    #     x_train, y_train = loader.to_lists(df_train, config)

    #     classifiers = build_classifiers()
    #     results = []

    #     for classifier in classifiers:
    #         print(f"Training {classifier.name}...")
    #         classifier.fit(x_train, y_train)
    #         y_pred = classifier.predict(x_test)
    #         metrics = evaluate_model(classifier.name, y_test, y_pred)
    #         results.append(metrics)

    #     results = sorted(results, key=lambda item: item["f1"], reverse=True)
    #     all_results[config.csv_path.name] = results
    # print_results_table(all_results)
    llama_cfg = ModelConfig(
        local_dir=Path("./classifier/llama_model/llama-3.1-8b"),
        device="cuda",
        load_in_4bit=True,
    )
    llama_train_cfg = TrainingConfig(
        epochs=1,            
        learning_rate=2e-4,     
        batch_size=4,           
        lora_r=8,             
        lora_alpha=16,          
        output_dir="checkpoints-qualified,1h",
    )
    classify_cfg = ClassifyConfig(
        threshold=0.5,
        temperature=0.1,
        max_new_tokens=32,
    )
    loader = CsvDataLoader()
    prompt_builder = LlamaSpamPromptBuilder()

    config_setup = config1
    raw_data = loader.load(config_setup)
    data = loader.preprocess(raw_data, config_setup)
    df_train, df_test = loader.split(raw_data, config_setup)
    llama = setup_model(llama_cfg)
    # Aby wytrenować model, odkomentuj poniższą linię:
    # train(llama, prompt_builder, df_train, df_test, llama_train_cfg, config_setup)
    
    classifier = LlamaSpamClassifier(llama, prompt_builder, classify_cfg)
    classifier.fit([], [], adapter_path=f"{llama_train_cfg.output_dir}/final")
    x_test, y_test = loader.to_lists(df_test, config_setup)
    sample = 90  # liczba próbek testowych używanych do oceny
    y_pred = classifier.predict(x_test[:sample])
    print(classification_report(y_test[:sample], y_pred, target_names=["ham", "spam"]))
if __name__ == "__main__":
    main()
