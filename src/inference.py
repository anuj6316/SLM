import os
import json
import torch
import argparse
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel

# 1. Setup Logging
# We use the logging module instead of print() because it allows us to
# differentiate between standard info, warnings, and errors easily.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, max_seq_length: int = 2048) -> Tuple[Any, Any]:
    """
    Loads the model and tokenizer using Unsloth's optimized loader.
    """
    logger.info(f"Initializing model loading from: {model_path}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,  # Automatic detection
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully and set to inference mode.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads data from either .json or .jsonl files.
    """
    logger.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(data_path)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            else:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        raise

def format_prompt(instruction: str, system_prompt: str) -> str:
    """
    Wraps the instruction in a ChatML-style prompt.
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_sql(full_output: str) -> str:
    """
    Cleans the model output to extract only the SQL query.
    Handles 'assistant' tags and markdown code blocks.
    """
    # 1. Split by assistant tag if present
    if "assistant\n" in full_output:
        prediction = full_output.split("assistant\n")[-1].strip()
    else:
        prediction = full_output.strip()

    # 2. Use regex to extract content from markdown blocks (e.g., ```sql ... ```)
    # This is more robust than simple .replace()
    sql_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", prediction, re.DOTALL | re.IGNORECASE)
    if sql_match:
        prediction = sql_match[1].strip()
    else:
        # Fallback: remove any leftover backticks if regex didn't find a block
        prediction = prediction.replace("```sql", "").replace("```", "").strip()

    # 3. Flatten to single line
    return " ".join(prediction.split())

def run_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    system_prompt: str = "You are a Text-to-SQL expert.",
    max_new_tokens: int = 128
):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running inference on device: {device}")

    # Load resources
    model, tokenizer = load_model_and_tokenizer(model_path)
    data = load_dataset(data_path)

    results = []
    logger.info(f"Processing {len(data)} items...")

    for item in tqdm(data, desc="Generating SQL"):
        # Flexibility for different field names
        instruction = item.get("instruction") or item.get("question", "")
        gold = item.get("output") or item.get("query", "N/A")
        
        prompt = format_prompt(instruction, system_prompt)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = extract_sql(full_output)

        results.append({
            "input": instruction,
            "gold_query": gold,
            "model_output": prediction
        })

    # Save Results
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data (.json or .jsonl)")
    parser.add_argument("--output_path", type=str, default="inference_results.json", help="Path to save results")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    try:
        run_inference(
            model_path=args.model_path,
            data_path=args.data_path,
            output_path=args.output_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens
        )
    except Exception as e:
        logger.critical(f"Inference pipeline crashed: {str(e)}")