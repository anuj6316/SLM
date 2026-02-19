import os
import json
import torch
import argparse
import subprocess
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel
from core.schemas import load_schema_dict, build_sft_prompt
from core.config import load_config, AppConfig

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

def evaluate_model(config: AppConfig):
    """
    Full evaluation pipeline using AppConfig.
    """
    eval_cfg = getattr(config, 'evaluation', config) # Fallback to global if no eval section
    output_dir = getattr(eval_cfg, 'output_dir', "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluation starting on device: {device}")

    # 1. Load Model
    model_path = getattr(eval_cfg, 'model_path', config.paths.model_output)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config.model.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

    # 2. Load Data & Schemas
    data_path = getattr(eval_cfg, 'data_path', config.paths.data_path)
    tables_path = getattr(eval_cfg, 'tables_path', None)
    try:
        schemas = load_schema_dict(tables_path) if tables_path else None
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        logger.info(f"Loaded {len(data)} evaluation examples.")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

    # 3. Generate Predictions
    gold_path = os.path.join(output_dir, "gold.txt")
    pred_path = os.path.join(output_dir, "pred.txt")
    
    logger.info("Generating SQL predictions...")
    
    with open(gold_path, "w", encoding="utf-8") as f_gold, \
         open(pred_path, "w", encoding="utf-8") as f_pred:
        
        for ex in tqdm(data, desc="Evaluating"):
            db_id = ex.get("db_id", "unknown")
            question = ex.get("question", "")
            gold_sql = ex.get("query", "")
            
            f_gold.write(f"{gold_sql}\t{db_id}\n")
            
            if schemas and db_id in schemas:
                instruction = build_sft_prompt(db_id, schemas[db_id], question)
            elif "instruction" in ex:
                instruction = ex["instruction"]
            else:
                instruction = f"Question: {question}"

            prompt = (
                f"<|im_start|>system\n{config.formatting.system_prompt}<|im_end|>\n"
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
            pred_sql = generated_text.split("assistant\n")[-1].strip() if "assistant\n" in generated_text else generated_text.strip()
            pred_sql = pred_sql.replace("```sql", "").replace("```", "").strip()
            pred_sql = " ".join(pred_sql.split())
            
            f_pred.write(f"{pred_sql}\n")

    # 4. Final Evaluation
    db_dir = getattr(eval_cfg, 'db_dir', None)
    spider_eval_path = getattr(eval_cfg, 'spider_eval_path', "data/spider/spider-master/evaluation.py")
    if tables_path and db_dir and os.path.exists(spider_eval_path):
        run_spider_evaluation(spider_eval_path, gold_path, pred_path, db_dir, tables_path)
    else:
        logger.info(f"Skipping official Spider evaluation. Results saved to {pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    app_config = load_config(args.config)
    evaluate_model(app_config)
