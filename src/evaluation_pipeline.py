import os
import json
import torch
import argparse
import subprocess
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel
from schema_utils import load_schema_dict, build_sft_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_spider_evaluation(
    spider_eval_path: str,
    gold_path: str,
    pred_path: str,
    db_dir: str,
    tables_path: str
):
    """Executes the official Spider evaluation script."""
    logger.info("Running official Spider evaluation...")
    cmd = [
        "python3", spider_eval_path,
        "--gold", gold_path,
        "--pred", pred_path,
        "--db", db_dir,
        "--table", tables_path,
        "--etype", "all"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Evaluation Output:\n" + result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation script failed with error:\n{e.stderr}")

def evaluate_model(
    model_path: str,
    data_path: str,
    tables_path: str,
    db_dir: str,
    output_dir: str,
    spider_eval_path: str = "data/spider/spider-master/evaluation.py"
):
    """
    Full evaluation pipeline: Loads model, generates SQL, and runs official scoring.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluation starting on device: {device}")

    # 1. Load Model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return

    # 2. Load Data & Schemas
    try:
        schemas = load_schema_dict(tables_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} evaluation examples.")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return

    # 3. Generate Predictions
    gold_path = os.path.join(output_dir, "gold.txt")
    pred_path = os.path.join(output_dir, "pred.txt")
    
    logger.info("Generating SQL predictions...")
    
    with open(gold_path, "w", encoding="utf-8") as f_gold, \
         open(pred_path, "w", encoding="utf-8") as f_pred:
        
        for ex in tqdm(data, desc="Evaluating"):
            db_id = ex["db_id"]
            question = ex["question"]
            gold_sql = ex["query"]
            
            # Save Gold entry
            f_gold.write(f"{gold_sql}\t{db_id}\n")
            
            # Prepare prompt using centralized logic
            instruction = build_sft_prompt(db_id, schemas[db_id], question)
            prompt = (
                f"<|im_start|>system\nYou are a Text-to-SQL expert.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract SQL (Post-processing)
            if "assistant\n" in generated_text:
                pred_sql = generated_text.split("assistant\n")[-1].strip()
            elif "SQL:" in generated_text:
                pred_sql = generated_text.split("SQL:")[-1].strip()
            else:
                pred_sql = generated_text.strip()
            
            # Clean up: remove markdown and normalize whitespace
            pred_sql = pred_sql.replace("```sql", "").replace("```", "").strip()
            pred_sql = " ".join(pred_sql.split())
            
            f_pred.write(f"{pred_sql}\n")

    # 4. Final Evaluation
    run_spider_evaluation(spider_eval_path, gold_path, pred_path, db_dir, tables_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spider Evaluation Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str, default="data/spider/dev.json")
    parser.add_argument("--tables_path", type=str, default="data/spider/tables.json")
    parser.add_argument("--db_dir", type=str, default="data/spider/database")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            tables_path=args.tables_path,
            db_dir=args.db_dir,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.critical(f"Evaluation pipeline crashed: {str(e)}")