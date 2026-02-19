import os
import torch
import argparse
import logging
import sys
from typing import Optional, Dict, Any
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from core.config import load_config, AppConfig

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def display_setup_dashboard(config: AppConfig):
    """Displays a beautiful dashboard of the training setup."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="magenta")

    table.add_row("Base Model", config.model.model_name)
    table.add_row("Max Seq Length", str(config.model.max_seq_length))
    table.add_row("Data Path", config.paths.data_path)
    table.add_row("Epochs", str(config.training.epochs))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Grad Accumulation", str(config.training.gradient_accumulation_steps))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Output Dir", config.training.output_dir)

    console.print(Panel(table, title="[bold blue]Training Configuration[/]", border_style="blue", expand=False))

def display_model_card(output_dir: str):
    """Displays a model card summary after training."""
    table = Table(show_header=False, box=None)
    table.add_row("[bold green]Status[/]", "Success ✅")
    table.add_row("[bold green]Location[/]", output_dir)
    table.add_row("[bold green]Format[/]", "PEFT / LoRA Adapters")
    table.add_row("[bold green]Deploy With[/]", "Inference.py")

    console.print("\n")
    console.print(Panel(table, title="[bold green]Model Card / Training Complete[/]", border_style="green", expand=False))

def train_model(config: AppConfig):
    """
    Fine-tunes a model on SFT data using the global AppConfig.
    """
    display_setup_dashboard(config)

    # 1. Load Model & Tokenizer
    with console.status("[bold cyan]Loading base model and tokenizer...") as status:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model.model_name,
                max_seq_length=config.model.max_seq_length,
                load_in_4bit=config.model.load_in_4bit,
                dtype=config.model.dtype,
            )
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to load base model: {str(e)}")
            logger.error(f"Failed to load base model: {str(e)}")
            raise

    # 2. Configure LoRA
    with console.status("[bold magenta]Injecting PEFT/LoRA adapters...") as status:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            target_modules=config.lora.target_modules,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # 3. Load and Format Dataset
    with console.status("[bold yellow]Preparing dataset and applying chat template...") as status:
        data_path = config.paths.data_path
        if not os.path.exists(data_path):
            console.print(f"[bold red]Error:[/] Dataset file not found: {data_path}")
            raise FileNotFoundError(data_path)

        dataset = load_dataset("json", data_files=data_path, split="train")

        from unsloth import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = config.formatting.chat_template,
            mapping = vars(config.formatting.mapping) if isinstance(config.formatting.mapping, AppConfig) else config.formatting.mapping,
        )

        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            if "messages" in example:
                return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
            elif "instruction" in example and "output" in example:
                messages = [
                    {"role": "system", "content": config.formatting.system_prompt},
                    {"role": "user", "content": example.get("instruction", "")},
                    {"role": "assistant", "content": example.get("output", "")}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            return {"text": ""}

        dataset = dataset.map(format_example)
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    # 4. Configure Trainer
    with console.status("[bold blue]Initializing SFTTrainer...") as status:
        sft_config = SFTConfig(
            output_dir=config.training.output_dir,
            max_seq_length=config.model.max_seq_length,
            dataset_text_field="text",
            packing=True,
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.training.logging_steps,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=config.training.save_steps or 500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            seed=config.training.seed,
            save_total_limit=2,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get('training').get('patience', 3))]
        )

    # 5. Run Training
    console.print(f"\n[bold green]🚀 Starting fine-tuning...[/]")
    trainer.train()

    # 6. Save Results
    with console.status("[bold green]Saving fine-tuned adapters...") as status:
        model.save_pretrained(config.training.output_dir)
        tokenizer.save_pretrained(config.training.output_dir)
    
    display_model_card(config.training.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    app_config = load_config(args.config)
    
    try:
        train_model(app_config)
    except Exception as e:
        console.print(f"[bold red]CRITICAL ERROR:[/] {str(e)}")
        logger.critical(f"Training pipeline crashed: {str(e)}")