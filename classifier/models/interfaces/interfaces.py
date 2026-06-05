from abc import ABC, abstractmethod
from typing import Optional

from classifier.models.classes import ClassifyConfig, DataConfig, ModelConfig, TrainingConfig


class IDataLoader(ABC):
    """Wczytywanie, podział i wstępne przetwarzanie danych tekstowych."""

    @abstractmethod
    def load(self, config: DataConfig):
        """Wczytuje CSV wskazany przez config.csv_path, zwraca DataFrame."""

    @abstractmethod
    def split(self, data, config: DataConfig):
        """Dzieli dane na (train, test), stratyfikując po config.label_column."""

    @abstractmethod
    def preprocess(self, data, config: DataConfig):
        """Czyści tekst i normalizuje etykiety do 'spam'/'ham'."""

    # @abstractmethod
    # def get_label_distribution(self, data, config: DataConfig) -> dict[str, int]:
    #     """Zwraca rozkład klas {'spam': N, 'ham': M}."""

    @abstractmethod
    def to_lists(self, data, config: DataConfig) -> tuple[list[str], list[int]]:
        """Konwertuje DataFrame na (texts, labels: spam=1/ham=0)."""


class IModel(ABC):
    """Operacje związane z obsługą modelu językowego."""

    @abstractmethod
    def load(self, config: ModelConfig) -> None:
        """Ładuje model z config.local_dir do pamięci GPU/CPU."""

    @abstractmethod
    def generate(self, prompt: str, config: ClassifyConfig) -> str:
        """Generuje odpowiedź używając config.temperature i config.max_new_tokens."""

    @abstractmethod
    def unload(self) -> None:
        """Zwalnia model z pamięci i czyści cache GPU."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Zwraca True jeśli model jest załadowany i gotowy."""


class IPromptBuilder(ABC):
    """Budowanie promptów i interpretacja odpowiedzi modelu."""

    @abstractmethod
    def build(self, text: str) -> str:
        """Buduje prompt klasyfikacyjny (bez oczekiwanej odpowiedzi)."""

    @abstractmethod
    def build_training_sample(self, text: str, label: str) -> str:
        """Buduje przykład treningowy: prompt + etykieta + token końca."""

    @abstractmethod
    def parse(self, response: str) -> str:
        """Parsuje odpowiedź modelu → 'spam', 'ham' lub 'unknown'."""

    @abstractmethod
    def build_batch(self, texts: list[str]) -> list[str]:
        """Wygodna wersja build() dla listy tekstów."""

    @abstractmethod
    def build_training_batch(self, texts: list[str], labels: list[str]) -> list[str]:
        """Wygodna wersja build_training_sample() dla list."""


class IClassifier(ABC):
    """Logika klasyfikacji wiadomości."""

    @abstractmethod
    def fit(self, x_train: list[str], y_train: list[int], **kwargs) -> None:
        """Inicjalizuje klasyfikator: ładuje model i/lub adaptery LoRA."""

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        """Zwraca przewidywane etykiety 0/1 dla zbioru tekstów."""

    @abstractmethod
    def classify(self, text: str) -> str:
        """Klasyfikuje pojedynczą wiadomość, zwraca 'spam' lub 'ham'."""

    @abstractmethod
    def classify_batch(self, texts: list[str]) -> list[str]:
        """Klasyfikuje listę wiadomości, zwraca etykiety słowne."""


class ITrainer(ABC):
    """Trenowanie i zapis modelu."""

    @abstractmethod
    def train(self, train_data, val_data, config: TrainingConfig) -> None:
        """Uruchamia fine-tuning używając config.epochs, learning_rate, batch_size, lora_*."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Zapisuje adaptery LoRA na dysk (nie cały model bazowy)."""

    @abstractmethod
    def evaluate(self) -> dict:
        """Ewaluuje model na zbiorze walidacyjnym, zwraca metryki."""
