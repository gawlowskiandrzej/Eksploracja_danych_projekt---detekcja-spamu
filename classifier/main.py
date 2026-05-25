import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from classifier.models.classes import DataConfig
from classifier.implementations.csv_data_loader import CsvDataLoader
from classifier.implementations.sklearn_classifier import SklearnClassifier


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
    config1 = DataConfig(
            csv_path=Path(f"data/spam_email_dataset.csv"),
            text_column="email_text",
            label_column="label",
            test_size=0.2,
            seed=42,
        )
    config2 = DataConfig(
            csv_path=Path(f"data/email_dataset_100k.csv"),
            text_column="body_plain",
            label_column="label",
            test_size=0.2,
            seed=42,
        )
    config3 = DataConfig(
            csv_path=Path(f"data/llm_spam_email_dataset.csv"),
            text_column="email_text",
            label_column="label",
            test_size=0.2,
            seed=42,
        )
    all_results = {}
    for config in [config1, config2, config3]:
        print(f"Evaluating classifiers on {config.csv_path.name}...")
        loader = CsvDataLoader()
        raw_data = loader.load(config)
        data = loader.preprocess(raw_data, config)
        x_train, x_test, y_train, y_test = loader.split(data, config)

        classifiers = build_classifiers()
        results = []

        for classifier in classifiers:
            print(f"Training {classifier.name}...")
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            metrics = evaluate_model(classifier.name, y_test, y_pred)
            results.append(metrics)

        results = sorted(results, key=lambda item: item["f1"], reverse=True)
        all_results[config.csv_path.name] = results
    print_results_table(all_results)

if __name__ == "__main__":
    main()
