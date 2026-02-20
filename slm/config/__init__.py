"""
Configuration module for SLM Text-to-SQL.

Public API:
    settings: Global settings instance loaded from config.yaml
    Settings: Main settings class for type annotations
"""

from slm.config.settings import (
    Settings,
    settings,
    ProjectSettings,
    MLflowSettings,
    DatasetConfig,
    DataSettings,
    ModelSettings,
    LoRASettings,
    TrainingSettings,
    EvaluationSettings,
    InferenceSettings,
    FormattingSettings,
)

__all__ = [
    "settings",
    "Settings",
    "ProjectSettings",
    "MLflowSettings",
    "DatasetConfig",
    "DataSettings",
    "ModelSettings",
    "LoRASettings",
    "TrainingSettings",
    "EvaluationSettings",
    "InferenceSettings",
    "FormattingSettings",
]
