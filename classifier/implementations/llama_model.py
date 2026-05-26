import gc
import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from classifier.models.classes import ClassifyConfig, ModelConfig
from classifier.models.interfaces.interfaces import IModel

logger = logging.getLogger(__name__)


class LlamaModel(IModel):
    """
    Obsługuje Llama 3.1 8B z opcjonalnym kwantyzowaniem 4-bit (QLoRA).

    ModelConfig.local_dir  — ścieżka lokalna lub nazwa z HuggingFace Hub
    ModelConfig.device     — 'cuda' | 'cpu' | 'auto'
    ModelConfig.load_in_4bit — kwantyzacja NF4 przez bitsandbytes
    """

    def __init__(self) -> None:
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------
    # IModel
    # ------------------------------------------------------------------

    def load(self, config: ModelConfig) -> None:
        """Ładuje tokenizer i model; stosuje kwantyzację NF4 gdy load_in_4bit=True."""
        model_path = str(config.local_dir)
        logger.info("Ładowanie modelu z: %s  (4-bit=%s)", model_path, config.load_in_4bit)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Llama 3 nie definiuje pad_token — ustaw na eos_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token    = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        bnb_config = self._build_bnb_config() if config.load_in_4bit else None

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            # device_map="auto" automatycznie rozkłada model na dostępne GPU/CPU;
            # gdy load_in_4bit=True jest wymagane przez bitsandbytes
            device_map="auto" if config.load_in_4bit else config.device,
            torch_dtype=torch.float16 if not config.load_in_4bit else None,
        )
        # Wyłącz KV-cache — wymagane przez gradient checkpointing podczas treningu
        self._model.config.use_cache = False
        logger.info("Model załadowany pomyślnie")

    def generate(self, prompt: str, config: ClassifyConfig) -> str:
        """
        Generuje odpowiedź modelu.
        Zwraca wyłącznie nowo wygenerowane tokeny (bez echa promptu).
        """
        self._require_loaded()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        # do_sample=False → greedy decoding, deterministyczne wyniki
        # temperature z ClassifyConfig steruje ewentualnym samplingiem
        do_sample = config.temperature < 1.0 and config.temperature > 0.0

        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            temperature=config.temperature if do_sample else None,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, generation_config=gen_config)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        """Zwalnia model z pamięci i czyści cache GPU."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Model zwolniony z pamięci")

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    # ------------------------------------------------------------------
    # Properties dla QLoRATrainer
    # ------------------------------------------------------------------

    @property
    def model(self) -> AutoModelForCausalLM:
        self._require_loaded()
        return self._model  # type: ignore[return-value]

    @property
    def tokenizer(self) -> AutoTokenizer:
        self._require_loaded()
        return self._tokenizer  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Prywatne
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bnb_config() -> BitsAndBytesConfig:
        """NF4 z podwójnym kwantyzowaniem — standard dla QLoRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # ~0.4 bpp oszczędności VRAM
        )

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise RuntimeError("Model nie jest załadowany. Wywołaj najpierw load(config).")
