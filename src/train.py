import os
import torch
import argparse
import logging
from typing import Optional, Dict, Any
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
):
    """
    Fine-tunes a model on Text-to-SQL data using Unsloth and SFTTrainer.
    """
    logger.info(f"Loading base model: {model_name}")
    
    # 1. Load Model & Tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,  # Efficient training
            dtype=None,
        )
    except Exception as e:
        logger.error(f"Failed to load base model: {str(e)}")
        return

    # 2. Configure LoRA
    logger.info("Configuring PEFT/LoRA adapters...")
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

    # 3. Load and Format Dataset
    logger.info(f"Loading dataset: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Dataset file not found: {data_path}")
        return

    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_example(example: Dict[str, str]) -> Dict[str, str]:
        """Formats each example into the ChatML format."""
        return {
            "text": (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
            )
        }

    logger.info("Applying ChatML formatting to dataset...")
    dataset = dataset.map(format_example)

    # 4. Configure Trainer
    logger.info("Initializing SFTTrainer configuration...")
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # 5. Run Training
    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    trainer.train()

    # 6. Save Results
    logger.info(f"Saving fine-tuned model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning Pipeline")
    
    # SageMaker-friendly path detection
    # SM_CHANNEL_TRAINING is the path where SageMaker downloads your S3 data
    # SM_MODEL_DIR is where SageMaker expects the final model to be saved for S3 upload
    sm_data_dir = os.environ.get("SM_CHANNEL_TRAINING")
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    
    default_data = os.path.join(sm_data_dir, "train_sft.jsonl") if sm_data_dir else "data/spider_sft/train_sft.jsonl"
    default_output = sm_model_dir if sm_model_dir else "outputs/qwen-text2sql"

    parser.add_argument("--data_path", type=str, default=default_data, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    
    args = parser.parse_args()
    
    try:
        train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            system_prompt=args.system_prompt,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.critical(f"Training pipeline crashed: {str(e)}")