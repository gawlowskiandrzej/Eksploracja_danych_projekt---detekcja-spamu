from classifier.models.classes import DataConfig
from pathlib import Path
from classifier.models.classes import ClassifyConfig,TrainingConfig, ModelConfig


main_dataset = DataConfig(
            csv_path=Path(f"data/qualified_main_dataset.csv"),
            text_column="email_text",
            label_column="label",
            separator=";",
            test_size=0.2,
            seed=42,
        )
second_dataset = DataConfig(
            csv_path=Path(f"data/qualified_second_dataset.csv"),
            text_column="email_text",
            label_column="label",
            separator=",",
            test_size=0.90,
            seed=42,
        )

llama_cfg = ModelConfig(
        local_dir=Path("./classifier/llama_model/llama-3.1-8b"),
        device="cuda",
        load_in_4bit=True,
    )
llama_train_cfg = TrainingConfig(
        epochs=1,            
        learning_rate=2e-4,     
        batch_size=4,           
        lora_r=8,             
        lora_alpha=16,          
        output_dir="checkpoints-qualified,1h",
    )
classify_cfg = ClassifyConfig(
        threshold=0.5,
        temperature=0.1,
        max_new_tokens=32,
    )