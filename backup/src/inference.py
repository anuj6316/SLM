import os
import json
import torch
import argparse
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel
from schema_utils import load_schema_dict, build_sft_prompt

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, max_seq_length: int = 2048) -> Tuple[Any, Any]:
    """Loads the model and tokenizer using Unsloth's optimized loader."""
    logger.info(f"Loading model from: {model_path}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def run_inference(
    model_path: str,
    data_path: str,
    tables_path: str,
    output_path: str,
    system_prompt: str = "You are a Text-to-SQL expert.",
    max_new_tokens: int = 128
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running inference on {device}")

    # 1. Load Resources
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    logger.info(f"Loading schemas from {tables_path}")
    schemas = load_schema_dict(tables_path)
    
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    logger.info(f"Processing {len(data)} items...")

    for item in tqdm(data, desc="Generating SQL"):
        db_id = item.get("db_id")
        question = item.get("question")
        gold = item.get("query", "N/A")

        if not db_id or not question:
            logger.warning("Skipping item: missing db_id or question.")
            continue

        # 2. Inject Schema (Crucial for Qwen/Spider)
        if db_id in schemas:
            # Use the centralized prompt builder to match training format
            instruction = build_sft_prompt(db_id, schemas[db_id], question)
        else:
            logger.warning(f"Schema for {db_id} not found. Using raw question.")
            instruction = f"Question: {question}\nSQL:"

        # 3. Format Prompt
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 4. Generate
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. Robust Extraction: Decode ONLY the new tokens
        new_tokens = outputs[0][input_length:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up markdown if model adds it
        prediction = re.sub(r"```(?:sql)?\s*(.*?)\s*```", r"\1", prediction, flags=re.DOTALL | re.IGNORECASE).strip()
        prediction = " ".join(prediction.split())

        results.append({
            "db_id": db_id,
            "question": question,
            "gold_query": gold,
            "model_output": prediction
        })

    # 6. Save Results
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Inference Script")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to dev.json")
    parser.add_argument("--tables_path", type=str, required=True, help="Path to tables.json")
    parser.add_argument("--output_path", type=str, default="inference_results.json")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--max_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    try:
        run_inference(
            model_path=args.model_path,
            data_path=args.data_path,
            tables_path=args.tables_path,
            output_path=args.output_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens
        )
    except Exception as e:
        logger.critical(f"Inference crashed: {str(e)}")
