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
from config_utils import load_config

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.columns import Columns

console = Console()

# Setup logging - Keep it minimal in console, detailed in file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def display_setup_dashboard(params: Dict[str, Any]):
    """Displays a beautiful dashboard of the training setup."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="magenta")

    table.add_row("Base Model", params.get("model_name"))
    table.add_row("Max Seq Length", str(params.get("max_seq_length")))
    table.add_row("Data Path", params.get("data_path"))
    table.add_row("Epochs", str(params.get("epochs")))
    table.add_row("Batch Size", str(params.get("batch_size")))
    table.add_row("Grad Accumulation", str(params.get("grad_accum")))
    table.add_row("Learning Rate", str(params.get("learning_rate")))
    table.add_row("Output Dir", params.get("output_dir"))

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

def train_model(
    data_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    system_prompt: str = "You are a Text-to-SQL expert.",
    max_seq_length: int = 2048,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    load_in_4bit: bool = True,
    r: int = 32,
    lora_alpha: int = 32,
    patience: int = 3,
):
    """
    Fine-tunes a model on SFT data using Unsloth and SFTTrainer with Early Stopping.
    """
    display_setup_dashboard({
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "data_path": data_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "output_dir": output_dir,
        "patience": patience
    })

    # 1. Load Model & Tokenizer
    with console.status("[bold cyan]Loading base model and tokenizer...") as status:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=None,
            )
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to load base model: {str(e)}")
            return

    # 2. Configure LoRA
    with console.status("[bold magenta]Injecting PEFT/LoRA adapters...") as status:
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # 3. Load and Format Dataset
    with console.status("[bold yellow]Preparing dataset and applying chat template...") as status:
        if not os.path.exists(data_path):
            console.print(f"[bold red]Error:[/] Dataset file not found: {data_path}")
            return

        dataset = load_dataset("json", data_files=data_path, split="train")

        from unsloth import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml",
            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
        )

        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            if "messages" in example:
                return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
            elif "instruction" in example and "output" in example:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example.get("instruction", "")},
                    {"role": "assistant", "content": example.get("output", "")}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            return {"text": ""}

        dataset = dataset.map(format_example)
        
        # Split into train and validation sets
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    # 4. Configure Trainer
    with console.status("[bold blue]Initializing SFTTrainer...") as status:
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            packing=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            warmup_steps=min(50, int(len(train_dataset) * 0.1)),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_total_limit=2,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
        )

    # 5. Run Training
    console.print(f"\n[bold green]🚀 Starting fine-tuning...[/]")
    trainer.train()

    # 6. Save Results
    with console.status("[bold green]Saving fine-tuned adapters...") as status:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    display_model_card(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning Pipeline")
    
    # SageMaker paths
    sm_data_dir = os.environ.get("SM_CHANNEL_TRAINING")
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    
    default_data = os.path.join(sm_data_dir, "train_sft.jsonl") if sm_data_dir else "data/train_sft.jsonl"
    default_output = sm_model_dir if sm_model_dir else "outputs/qwen-text2sql"

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--patience", type=int)
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Correct resolution of params
    paths_config = config.get("paths", {})
    train_config = config.get("training", {})
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    format_config = config.get("formatting", {})

    data_path = args.data_path or paths_config.get("data_path") or default_data
    output_dir = args.output_dir or train_config.get("output_dir") or default_output
    model_name = args.model_name or model_config.get("model_name") or "Qwen/Qwen2.5-1.5B-Instruct"
    system_prompt = format_config.get("system_prompt") or "You are a Text-to-SQL expert."
    
    epochs = args.epochs or train_config.get("epochs") or 2
    batch_size = args.batch_size or train_config.get("per_device_train_batch_size") or 2
    grad_accum = args.grad_accum or train_config.get("gradient_accumulation_steps") or 8
    learning_rate = args.learning_rate or train_config.get("learning_rate") or 2e-4
    max_seq_length = args.max_seq_length or model_config.get("max_seq_length") or 2048
    load_in_4bit = model_config.get("load_in_4bit", True)
    patience = args.patience or train_config.get("patience") or 3
    
    r = lora_config.get("r", 32)
    lora_alpha = lora_config.get("lora_alpha", 32)

    try:
        train_model(
            data_path=data_path,
            output_dir=output_dir,
            model_name=model_name,
            system_prompt=system_prompt,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            r=r,
            lora_alpha=lora_alpha,
            patience=patience
        )
    except Exception as e:
        console.print(f"[bold red]CRITICAL ERROR:[/] {str(e)}")
        logger.critical(f"Training pipeline crashed: {str(e)}")
