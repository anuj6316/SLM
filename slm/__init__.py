"""
SLM Text-to-SQL - Unified package for Text-to-SQL with Small Language Models.

Modules:
    - config: Pydantic settings configuration
    - data: Data pipeline (gather, clean, format, split)
    - training: SFT and GRPO training
    - evaluation: Benchmark evaluation
    - inference: Local and MLflow-traced inference
    - utils: Shared utilities
    - cli: Command-line interface

Quick Start:
    from slm.config import settings
    from slm.data import DataPipeline
    from slm.training import SFTTrainer
    from slm.inference import InferenceEngine

    # Data pipeline
    pipeline = DataPipeline(settings)
    pipeline.run_all()

    # Training
    trainer = SFTTrainer(settings)
    trainer.train()

    # Inference
    engine = InferenceEngine(settings)
    result = engine.generate("How many singers?", "concert_singer")
"""

from slm.config import settings, Settings
from slm.data import DataPipeline, SQLCleaner
from slm.training import SFTTrainer
from slm.evaluation import Evaluator
from slm.inference import InferenceEngine

__version__ = "1.0.0"

__all__ = [
    "settings",
    "Settings",
    "DataPipeline",
    "SQLCleaner",
    "SFTTrainer",
    "Evaluator",
    "InferenceEngine",
]
