import logging
from pathlib import Path
from typing import Optional



from classifier.models.classes import ClassifyConfig, ModelConfig
from classifier.implementations.llama_model import LlamaModel
from classifier.models.interfaces.interfaces import IClassifier
from classifier.implementations.prompt_builder import LlamaSpamPromptBuilder

from peft import PeftModel
logger = logging.getLogger(__name__)

_LABEL_TO_INT = {"spam": 1, "ham": 0, "unknown": -1}
_INT_TO_LABEL = {1: "spam", 0: "ham", -1: "unknown"}


class LlamaSpamClassifier(IClassifier):
    """
    Klasyfikator spamu oparty na Llama 3.1 8B + opcjonalne adaptery LoRA.

    fit() — "instaluje" model (ładuje wagi bazowe i/lub adaptery LoRA).
            Faktyczny trening odbywa się w QLoRATrainer.
    predict() — zwraca listę int: spam=1, ham=0, unknown=-1.
    classify() / classify_batch() — wygodne wersje zwracające etykiety słowne.

    ClassifyConfig.threshold — używany gdy model zwraca token 'unknown'
                               (fallback: klasyfikuj jako ham gdy < 0.5).
    """

    def __init__(
        self,
        llama_model: LlamaModel,
        prompt_builder: LlamaSpamPromptBuilder,
        classify_config: Optional[ClassifyConfig] = None,
    ) -> None:
        self._llama = llama_model
        self._prompt_builder = prompt_builder
        self._config = classify_config or ClassifyConfig()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # IClassifier
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: list[str],
        y_train: list[int],
        *,
        model_config: Optional[ModelConfig] = None,
        adapter_path: Optional[str] = None,
    ) -> None:
        """
        Przygotowuje klasyfikator do predykcji.

        model_config  — jeśli podany i model nie jest załadowany, ładuje go
        adapter_path  — ścieżka do wytrenowanych adapterów LoRA (opcjonalna)
        x_train/y_train — ignorowane (trening → QLoRATrainer)
        """
        if model_config is not None and not self._llama.is_loaded():
            self._llama.load(model_config)

        if adapter_path is not None:
            self._load_adapters(adapter_path)

        if not self._llama.is_loaded():
            raise RuntimeError(
                "Model nie jest załadowany. Podaj model_config lub załaduj "
                "model ręcznie przed wywołaniem fit()."
            )

        self._is_fitted = True
        logger.info("Klasyfikator gotowy do predykcji")

    def predict(self, texts: list[str]) -> list[int]:
        """
        Klasyfikuje listę tekstów.
        Zwraca: spam=1, ham=0.
        Gdy model zwróci niejednoznaczną odpowiedź, stosowany jest config.threshold jako fallback.
        """
        self._require_fitted()
        predictions: list[int] = []
 
        for i, text in enumerate(texts):
            prompt    = self._prompt_builder.build(text)
            response  = self._llama.generate(prompt, self._config)
            label_int = self._prompt_builder.parse(response)
 
            if label_int is None:
                # Fallback oparty na threshold — domyślnie ham (0) gdy threshold >= 0.5
                label_int = 1 if self._config.threshold < 0.5 else 0
                logger.debug("Fallback threshold dla tekstu %d → %d", i, label_int)
 
            predictions.append(label_int)
 
            if (i + 1) % 100 == 0:
                logger.info("Sklasyfikowano %d / %d", i + 1, len(texts))
 
        return predictions

    def classify(self, text: str) -> str:
        """Klasyfikuje pojedynczą wiadomość, zwraca 'spam' lub 'ham'."""
        return _INT_TO_LABEL[self.predict([text])[0]]

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Klasyfikuje listę wiadomości, zwraca etykiety słowne."""
        return [_INT_TO_LABEL[p] for p in self.predict(texts)]

    # ------------------------------------------------------------------
    # Prywatne
    # ------------------------------------------------------------------

    def _load_adapters(self, adapter_path: str) -> None:
        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Ścieżka do adapterów nie istnieje: {path}")
        self._llama._model = PeftModel.from_pretrained(
            self._llama.model,
            str(path),
            is_trainable=False,
        )
        logger.info("Adaptery LoRA załadowane z: %s", path)

    def _require_fitted(self) -> None:
        if not self._is_fitted or not self._llama.is_loaded():
            raise RuntimeError(
                "Klasyfikator nie jest gotowy. Wywołaj fit() z model_config "
                "lub załaduj model ręcznie."
            )
