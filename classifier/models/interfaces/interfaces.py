from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from classifier.models.classes import ClassifyConfig, DataConfig, ModelConfig, TrainingConfig

class IDataLoader(ABC):
    """
    Interfejs odpowiedzialny za wczytywanie, podział
    i wstępne przetwarzanie danych tekstowych.
    """
    @abstractmethod
    def load(self, config: DataConfig):
        """Wczytuje CSV, zwraca DataFrame."""

    @abstractmethod
    def split(self, data, config: DataConfig):
        """Dzieli dane na (train, test), stratyfikując po etykiecie."""

    @abstractmethod
    def preprocess(self, data, config: DataConfig):
        """Czyści tekst i normalizuje etykiety do 'spam'/'ham'."""


class IModel(ABC):
    """
    Interfejs definiujący operacje związane
    z obsługą modelu językowego.
    """
    @abstractmethod
    def load(self, config: ModelConfig) -> None:
        """Ładuje model i tokenizer z dysku do pamięci GPU/CPU."""

    @abstractmethod
    def generate(self, prompt: str, config: ClassifyConfig) -> str:
        """Generuje odpowiedź modelu dla podanego promptu."""

    @abstractmethod
    def unload(self) -> None:
        """Zwalnia model z pamięci i czyści cache GPU."""


class IPromptBuilder(ABC):
    """
    Interfejs odpowiedzialny za budowanie promptów
    i interpretację odpowiedzi modelu.
    """

    @abstractmethod
    def build(self, text: str) -> str:
        """Buduje prompt klasyfikacyjny dla podanej wiadomości."""

    @abstractmethod
    def build_training_sample(self, text: str, label: str) -> str:
        """Buduje przykład treningowy: prompt + oczekiwana odpowiedź."""

    @abstractmethod
    def parse(self, response: str) -> str:
        """Parsuje odpowiedź modelu, zwraca etykietę klasyfikacji ('spam' lub 'ham')."""


class IClassifier(ABC):
    """
    Interfejs definiujący logikę klasyfikacji wiadomości.
    """

    @abstractmethod
    def classify(self, text: str, config: Optional[ClassifyConfig] = None) -> str:
        """Klasyfikuje pojedynczą wiadomość tekstową
        i zwraca przewidywaną etykietę."""

    @abstractmethod
    def classify_batch(self, texts: list[str], config: Optional[ClassifyConfig] = None) -> list[str]:
        """Klasyfikuje listę wiadomości i zwraca listę
        odpowiadających im etykiet."""


class ITrainer(ABC):
    """
    Interfejs odpowiedzialny za trenowanie
    i zapis modelu.
    """
    @abstractmethod
    def train(self, train_data, val_data, config: TrainingConfig) -> None:
        """Uruchamia fine-tuning modelu na danych treningowych."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Zapisuje wytrenowany model na dysk."""
