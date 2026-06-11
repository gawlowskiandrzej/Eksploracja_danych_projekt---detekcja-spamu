import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from classifier.implementations.csv_data_loader import CsvDataLoader
from classifier.implementations.sklearn_classifier import SklearnClassifier
from config import main_dataset


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

def main():
    all_results = {}
    for config in [main_dataset]:
        print(f"Evaluating classifiers on {config.csv_path.name}...")
        loader = CsvDataLoader()
        raw_data = loader.load(config)
        data = loader.preprocess(raw_data, config)
        df_train, df_test = loader.split(data, config)
        x_test, y_test = loader.to_lists(df_test, config)
        x_train, y_train = loader.to_lists(df_train, config)

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