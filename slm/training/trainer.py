"""
SFT Trainer - Supervised Fine-Tuning for Text-to-SQL models.

Usage:
    from slm.training import SFTTrainer
    from slm.config import settings

    trainer = SFTTrainer(settings)
    trainer.train()
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from slm.config import Settings

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for Text-to-SQL models.

    Uses Unsloth for efficient LoRA training.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def setup(self) -> None:
        """Load model and tokenizer with LoRA adapters."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")

        logger.info(f"Loading model: {self._settings.model.name}")

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._settings.model.name,
            max_seq_length=self._settings.model.max_seq_length,
            dtype=self._settings.model.dtype,
            load_in_4bit=self._settings.model.load_in_4bit,
        )

        logger.info("Adding LoRA adapters...")
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self._settings.lora.r,
            target_modules=self._settings.lora.target_modules,
            lora_alpha=self._settings.lora.lora_alpha,
            lora_dropout=self._settings.lora.lora_dropout,
            bias=self._settings.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=self._settings.training.seed,
            use_rslora=False,
            loftq_config=None,
        )

    def _format_example(self, example: dict) -> str:
        """Format a training example to ChatML format."""
        system_prompt = self._settings.formatting.system_prompt
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

    def _prepare_dataset(self) -> tuple:
        """Load and format training dataset."""
        train_path = self._settings.data.train_file
        val_path = self._settings.data.val_file

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        logger.info(f"Loading dataset from {train_path}")

        dataset = load_dataset("json", data_files={"train": str(train_path)})

        if val_path.exists():
            dataset["validation"] = load_dataset(
                "json", data_files={"val": str(val_path)}
            )["train"]

        def formatting_func(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                text = self._format_example(
                    {
                        "instruction": examples["instruction"][i],
                        "input": examples["input"][i],
                        "output": examples["output"][i],
                    }
                )
                texts.append(text)
            return texts

        return dataset, formatting_func

    def train(self) -> Path:
        """
        Run SFT training.

        Returns:
            Path to output directory with trained model
        """
        self.setup()

        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
            from trl import SFTTrainer as TRISFTTrainer
            from transformers import TrainingArguments
        except ImportError as e:
            raise ImportError(f"Missing dependencies: {e}")

        dataset, formatting_func = self._prepare_dataset()

        self._tokenizer = get_chat_template(
            self._tokenizer,
            chat_template=self._settings.formatting.chat_template,
        )

        output_dir = Path(self._settings.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self._settings.training.epochs,
            per_device_train_batch_size=self._settings.training.per_device_train_batch_size,
            gradient_accumulation_steps=self._settings.training.gradient_accumulation_steps,
            learning_rate=self._settings.training.learning_rate,
            lr_scheduler_type=self._settings.training.lr_scheduler_type,
            warmup_steps=self._settings.training.warmup_steps,
            weight_decay=self._settings.training.weight_decay,
            seed=self._settings.training.seed,
            logging_steps=self._settings.training.logging_steps,
            save_steps=self._settings.training.save_steps,
            eval_strategy=self._settings.training.eval_strategy,
            eval_steps=self._settings.training.eval_steps,
            save_total_limit=self._settings.training.save_total_limit,
            optim=self._settings.training.optim,
            report_to=self._settings.training.report_to,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        )

        self._trainer = TRISFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            formatting_func=formatting_func,
            args=training_args,
        )

        logger.info("Starting training...")
        self._trainer.train()

        logger.info(f"Saving model to {output_dir}")
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        logger.info("Training complete!")
        return output_dir

    def save_merged(self, output_path: Optional[Path] = None) -> Path:
        """
        Save merged model (base + LoRA) for inference.

        Args:
            output_path: Path to save merged model (default: outputs/merged)

        Returns:
            Path to merged model
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call setup() or train() first.")

        output_path = output_path or Path(self._settings.training.output_dir) / "merged"
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from unsloth import FastLanguageModel

            self._model.save_pretrained_merged(
                str(output_path),
                self._tokenizer,
                save_method="merged_16bit",
            )
            logger.info(f"Merged model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")

        return output_path
