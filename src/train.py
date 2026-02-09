import os
import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from src.utils import get_logger

logger = get_logger(__name__)

def train_model(
    data_path,
    output_dir,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    system_prompt="You are a Text-to-SQL expert.",
    max_seq_length=2048,
    epochs=2,
    batch_size=2,
    grad_accum=8,
    learning_rate=2e-4,
):
    logger.info(f"Initializing model loading: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Set to True if memory is tight
        load_in_8bit=False,
        dtype=None,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    logger.info(f"Loading dataset from: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_example(example):
        return {
            "text": (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
            )
        }

    logger.info("Formatting dataset...")
    dataset = dataset.map(format_example)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=50,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=output_dir,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=True,
    )

    logger.info(f"Starting training for {epochs} epochs...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--epochs", type=int, default=2)
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        epochs=args.epochs
    )
