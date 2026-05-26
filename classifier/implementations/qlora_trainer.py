import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from classifier.models.classes import TrainingConfig
from classifier.models.interfaces.interfaces import ITrainer
from classifier.implementations.llama_model import LlamaModel
from classifier.implementations.prompt_builder import LlamaSpamPromptBuilder
from classifier.models.classes import DataConfig

logger = logging.getLogger(__name__)


class QLoRATrainer(ITrainer):
    """
    Fine-tuning Llama 3.1 8B za pomocą QLoRA (4-bit NF4 + PEFT LoRA).

    TrainingConfig mapowanie:
        epochs          → num_train_epochs
        learning_rate   → learning_rate
        batch_size      → per_device_train_batch_size
        lora_r          → LoraConfig.r
        lora_alpha      → LoraConfig.lora_alpha
        output_dir      → SFTConfig.output_dir

    Przykład:
        trainer = QLoRATrainer(llama_model, prompt_builder)
        trainer.train(df_train, df_val, config)
        trainer.save("checkpoints/final")
    """

    # Warstwy projekcji Llama 3 do fine-tuningu przez LoRA
    _DEFAULT_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        llama_model: LlamaModel,
        prompt_builder: LlamaSpamPromptBuilder,
    ) -> None:
        self._llama = llama_model
        self._prompt_builder = prompt_builder
        self._trainer: Optional[SFTTrainer] = None

    # ------------------------------------------------------------------
    # ITrainer
    # ------------------------------------------------------------------

    def train(self, train_data, val_data, config: TrainingConfig, data_config: DataConfig) -> None:
        """
        Uruchamia QLoRA fine-tuning.

        train_data / val_data: pd.DataFrame z kolumnami wskazanymi przez DataConfig
                               (po wywołaniu loader.preprocess — 'text' i 'label')
        """
        logger.info("Przygotowanie modelu do treningu QLoRA …")

        model    = self._llama.model
        tokenizer = self._llama.tokenizer

        # 1. Gradient checkpointing + cast niekvantyzowanych warstw do fp32
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

        # 2. Adaptery LoRA z parametrami z TrainingConfig
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            target_modules=self._DEFAULT_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # 3. Datasety HuggingFace (każdy wiersz = gotowy prompt treningowy)
        train_dataset = self._build_dataset(train_data, data_config)
        val_dataset   = self._build_dataset(val_data, data_config)

        # 4. SFTConfig — używa bezpośrednio pól z TrainingConfig
        sft_cfg = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            # Efektywny batch = batch_size * gradient_accumulation_steps
            # Przy batch_size=4 i acc=4 → efektywnie 16 przykładów/krok
            gradient_accumulation_steps=4,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.001,
            fp16=False,                                            
            bf16=torch.cuda.is_available(), 
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="paged_adamw_8bit",   # zoptymalizowany pod QLoRA
            report_to="none",
            dataset_text_field="text"
        )

        self._trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logger.info("Rozpoczynam trening …")
        result = self._trainer.train()
        logger.info(
            "Trening zakończony | loss=%.4f | steps=%d",
            result.training_loss,
            result.global_step,
        )
        self._trainer.log_metrics("train", result.metrics)
        self._trainer.save_metrics("train", result.metrics)

    def save(self, path: str) -> None:
        """
        Zapisuje adaptery LoRA i tokenizer na dysk.
        Nie zapisuje całego modelu bazowego — wczytaj go osobno przez LlamaModel.load().
        """
        if self._trainer is None:
            raise RuntimeError("Wywołaj najpierw train() przed save().")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._trainer.model.save_pretrained(str(out))
        self._llama.tokenizer.save_pretrained(str(out))
        logger.info("Adaptery LoRA zapisane w: %s", out)

    def evaluate(self) -> dict:
        """Ewaluuje model na zbiorze walidacyjnym, zwraca słownik metryk."""
        if self._trainer is None:
            raise RuntimeError("Wywołaj najpierw train().")
        metrics = self._trainer.evaluate()
        self._trainer.log_metrics("eval", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prywatne
    # ------------------------------------------------------------------

    def _build_dataset(self, df, config: DataConfig) -> Dataset:
        """
        Konwertuje DataFrame (text, label) na Dataset z gotowymi promptami.
        Kolumna 'label' moze byc stringiem ('spam'/'ham') lub intem (1/0).
        """
        _WORD_TO_INT = {"spam": 1, "ham": 0}
        samples = []
        for _, row in df.iterrows():
            label = row[config.label_column]
            if isinstance(label, str):
                label = _WORD_TO_INT[label.strip().lower()]
            samples.append(self._prompt_builder.build_training_sample(row[config.text_column], int(label)))
        return Dataset.from_dict({"text": samples})
