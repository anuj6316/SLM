"""
Training Module - SFT and GRPO training for Text-to-SQL.

Public API:
    SFTTrainer: Supervised Fine-Tuning trainer using Unsloth + LoRA
"""

from slm.training.trainer import SFTTrainer

__all__ = ["SFTTrainer"]
