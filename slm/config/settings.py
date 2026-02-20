"""
Configuration settings using Pydantic.

Usage:
    from slm.config import settings
    print(settings.model.name)
    print(settings.training.learning_rate)
"""

from pathlib import Path
from typing import List, Optional, Any, Dict

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Project metadata."""

    name: str = "SLM Text-to-SQL"
    version: str = "1.0.0"
    description: str = "Small Language Model for Text-to-SQL"


class MLflowSettings(BaseSettings):
    """MLflow tracking configuration."""

    enabled: bool = True
    tracking_uri: str = "databricks"
    experiment_name: str = "/Shared/text2sql"
    registry_uri: str = "databricks-uc"


class DatasetConfig(BaseSettings):
    """Individual dataset configuration."""

    name: str = ""
    source: str = ""
    split: str = "train"


class DataSettings(BaseSettings):
    """Data pipeline configuration."""

    datasets: List[DatasetConfig] = Field(default_factory=list)
    output_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    train_file: Path = Path("data/train_split.jsonl")
    val_file: Path = Path("data/val_split.jsonl")
    tables_file: Path = Path("data/raw/spider_tables.json")
    train_split: float = 0.95
    seed: int = 42

    @field_validator(
        "output_dir", "raw_dir", "train_file", "val_file", "tables_file", mode="before"
    )
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        return Path(v) if isinstance(v, str) else v


class ModelSettings(BaseSettings):
    """Model configuration."""

    name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None


class LoRASettings(BaseSettings):
    """LoRA/PEFT configuration."""

    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingSettings(BaseSettings):
    """Training configuration."""

    output_dir: Path = Path("outputs/qwen-coder-3b-text2sql")
    epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 0.01
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 2
    optim: str = "adamw_8bit"
    report_to: str = "mlflow"

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        return Path(v) if isinstance(v, str) else v


class EvaluationSettings(BaseSettings):
    """Evaluation configuration."""

    benchmark: str = "spider"
    output_dir: Path = Path("eval_results")
    batch_size: int = 1

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        return Path(v) if isinstance(v, str) else v


class InferenceSettings(BaseSettings):
    """Inference configuration."""

    data_path: Path = Path("data/val_split.jsonl")
    max_new_tokens: int = 128
    output_path: Optional[Path] = None
    num_samples: Optional[int] = None
    do_sample: bool = False
    temperature: float = 0.0

    @field_validator("data_path", "output_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Optional[Path]:
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class FormattingSettings(BaseSettings):
    """Prompt formatting configuration."""

    system_prompt: str = (
        "You are an expert Text-to-SQL assistant. "
        "Convert the natural language question into a valid SQL query based on the schema."
    )
    chat_template: str = "chatml"


class Settings(BaseSettings):
    """Main settings class combining all configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    lora: LoRASettings = Field(default_factory=LoRASettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    formatting: FormattingSettings = Field(default_factory=FormattingSettings)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config: Dict[str, Any] = yaml.safe_load(f) or {}

        def convert_dicts_to_models(data: Dict[str, Any]) -> Dict[str, Any]:
            """Convert nested dicts to their respective model classes."""
            result = {}
            for key, value in data.items():
                if key == "project" and isinstance(value, dict):
                    result[key] = ProjectSettings(**value)
                elif key == "mlflow" and isinstance(value, dict):
                    result[key] = MLflowSettings(**value)
                elif key == "data" and isinstance(value, dict):
                    if "datasets" in value:
                        value["datasets"] = [
                            DatasetConfig(**d) for d in value["datasets"]
                        ]
                    result[key] = DataSettings(**value)
                elif key == "model" and isinstance(value, dict):
                    result[key] = ModelSettings(**value)
                elif key == "lora" and isinstance(value, dict):
                    result[key] = LoRASettings(**value)
                elif key == "training" and isinstance(value, dict):
                    result[key] = TrainingSettings(**value)
                elif key == "evaluation" and isinstance(value, dict):
                    result[key] = EvaluationSettings(**value)
                elif key == "inference" and isinstance(value, dict):
                    result[key] = InferenceSettings(**value)
                elif key == "formatting" and isinstance(value, dict):
                    result[key] = FormattingSettings(**value)
                else:
                    result[key] = value
            return result

        converted = convert_dicts_to_models(yaml_config)
        return cls(**converted)


settings = Settings.from_yaml()
