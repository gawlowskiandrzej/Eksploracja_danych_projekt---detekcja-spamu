import pandas as pd
from sklearn.model_selection import train_test_split

from classifier.models.interfaces.interfaces import IDataLoader
from classifier.models.classes import DataConfig


class CsvDataLoader(IDataLoader):
    """
    Klasa odpowiedzialna za wczytywanie danych z pliku CSV, wstępne przetwarzanie
    tekstu i podział na zbiory treningowy i testowy.
    """
    def load(self, config: DataConfig):
        df = pd.read_csv(config.csv_path, sep=config.separator)
        return df

    def preprocess(self, data, config: DataConfig):
        df = data.copy()
        if (
            config.text_column not in df.columns
            or config.label_column not in df.columns
        ):
            raise ValueError(
                f"CSV musi zawierać kolumny '{config.text_column}' i '{config.label_column}'"
            )

        df = df[[config.text_column, config.label_column]].rename(
            columns={config.text_column: "text", config.label_column: "label"}
        )
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = (
            df["label"].astype(str).str.strip().replace({"spam": "1", "ham": "0"})
        )
        df = df[df["text"].ne("")].copy()
        df["label"] = df["label"].map({"1": 1, "0": 0})
        if df["label"].isna().any():
            raise ValueError(
                "Nieprawidłowe etykiety w zbiorze danych. Oczekiwano 0/1, spam/ham."
            )
        return df

    def split(self, data, config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Zwraca (df_train, df_test) — oba jako DataFrame z oryginalnymi kolumnami."""
        train, test = train_test_split(
            data,
            test_size=config.test_size,
            random_state=config.seed,
            stratify=data[config.label_column],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)
    def to_lists(self, data: pd.DataFrame, config: DataConfig) -> tuple[list[str], list[int]]:
        """
        Konwertuje DataFrame na (texts, labels: spam=1, ham=0).
        Obsługuje etykiety jako int (1/0) i string ('spam'/'ham').
        """
        texts = data[config.text_column].tolist()
        col = data[config.label_column]
        if col.dtype == object:
            labels = col.map({"spam": 1, "ham": 0}).astype(int).tolist()
        else:
            labels = col.astype(int).tolist()
        return texts, labels
