from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from classifier.models.interfaces.interfaces import IClassifier
from classifier.models.classes import ClassifyConfig


class SklearnClassifier(IClassifier):
    """
    Klasa implementująca klasyfikator tekstu za pomocą modeli z biblioteki scikit-learn.
    Każdy model jest opakowany w pipeline z TfidfVectorizer do przetwarzania tekstu.
    """
    def __init__(self, name: str, classifier: object):
        self.name = name
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=25000)),
                ("clf", classifier),
            ]
        )

    def fit(self, x_train: list[str], y_train: list[int]) -> None:
        self.pipeline.fit(x_train, y_train)

    def predict(self, texts: list[str]) -> list[int]:
        predictions = self.pipeline.predict(texts)
        return [int(pred) for pred in predictions]

    # def classify(self, text: str, config: Optional[ClassifyConfig] = None) -> str:
    #     prediction = self.pipeline.predict([text])[0]
    #     return "spam" if int(prediction) == 1 else "ham"

    # def classify_batch(
    #     self, texts: list[str], config: Optional[ClassifyConfig] = None
    # ) -> list[str]:
    #     predictions = self.pipeline.predict(texts)
    #     return ["spam" if int(pred) == 1 else "ham" for pred in predictions]
