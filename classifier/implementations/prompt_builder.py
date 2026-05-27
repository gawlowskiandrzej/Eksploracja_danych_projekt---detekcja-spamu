import re
import logging

from classifier.models.interfaces.interfaces import IPromptBuilder

_INT_TO_WORD: dict[int, str] = {1: "spam", 0: "ham"}
_WORD_TO_INT: dict[str, int] = {"spam": 1, "ham": 0}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Szablony promptów w formacie Llama 3 chat
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a spam detection classifier. "
    "Your task is to classify messages as 'spam' or 'ham'. "
    "Respond with ONLY the single word: spam or ham. "
    "Do not explain, do not add punctuation."
)

_CLASSIFICATION_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Classify this message:\n{text}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

_TRAINING_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Classify this message:\n{text}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{label}"
    "<|eot_id|>"
)

_VALID_LABELS = {"spam", "ham"}


class LlamaSpamPromptBuilder(IPromptBuilder):
    """
    Buduje prompty w formacie Llama 3 Instruct (<|...|> tokeny chat).
    Kompatybilny z meta-llama/Meta-Llama-3.1-8B-Instruct.
    """

    def __init__(
        self,
        system_prompt: str = _SYSTEM_PROMPT,
        max_text_chars: int = 1024,
    ) -> None:
        self._system = system_prompt
        self._max_text_chars = max_text_chars

    def build(self, text: str) -> str:
        """
        Buduje prompt klasyfikacyjny (bez oczekiwanej odpowiedzi).
        Model powinien wygenerować wyłącznie 'spam' lub 'ham'.
        """
        return _CLASSIFICATION_TEMPLATE.format(
            system=self._system,
            text=self._truncate(text),
        )

    def build_training_sample(self, text: str, label: int) -> str:
        """
        Buduje kompletny przykład treningowy używany przez ITrainer.
        Label: 1 = spam, 0 = ham.
        """
        if label not in _INT_TO_WORD:
            raise ValueError(f"Nieprawidłowa etykieta: {label!r}. Oczekiwano 0 lub 1.")
        word_label = _INT_TO_WORD[label]
        return _TRAINING_TEMPLATE.format(
            system=self._system,
            text=self._truncate(text),
            label=word_label,
        )

    def parse(self, response: str) -> int | None:
        """
        Parsuje odpowiedź modelu.
        Zwraca: 1 (spam), 0 (ham) lub None gdy odpowiedź niejednoznaczna.
        """
        cleaned = response.strip().lower()
        cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned).strip()
 
        first_word = cleaned.split()[0] if cleaned.split() else ""
        if first_word in _WORD_TO_INT:
            return _WORD_TO_INT[first_word]
 
        for word, int_label in _WORD_TO_INT.items():
            if word in cleaned:
                logger.debug("Etykieta '%s' znaleziona heurystycznie w: %r", word, response)
                return int_label
 
        logger.warning("Nie można sparsować odpowiedzi modelu: %r", response)
        return None

    def build_batch(self, texts: list[str]) -> list[str]:
        """Wygodna metoda budująca listę promptów na raz."""
        return [self.build(t) for t in texts]

    def build_training_batch(
        self, texts: list[str], labels: list[str]
    ) -> list[str]:
        """Wygodna metoda budująca listę przykładów treningowych."""
        if len(texts) != len(labels):
            raise ValueError("texts i labels muszą mieć tę samą długość")
        return [self.build_training_sample(t, l) for t, l in zip(texts, labels)]

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_text_chars:
            logger.debug("Tekst obcięty z %d do %d znaków", len(text), self._max_text_chars)
            return text[: self._max_text_chars]
        return text
