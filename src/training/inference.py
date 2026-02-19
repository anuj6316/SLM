import os
import json
import torch
import argparse
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel
from core.schemas import load_schema_dict, build_sft_prompt
from core.config import load_config, AppConfig

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_inference(config: AppConfig):
    inf_cfg = getattr(config, 'inference', config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running inference on {device}")

    # 1. Load Model
    # CLI args override config if set
    model_id = getattr(inf_cfg, 'model_id', getattr(config.paths, 'model_output', None))
    if not model_id:
        raise ValueError("Model ID or path must be specified via --model_id or config.")

    logger.info(f"Loading model from: {model_id}")

    # Check if model_id is a path or HF ID
    # FastLanguageModel.from_pretrained handles both
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=False, # Use False for inference usually, or config setting
    )
    FastLanguageModel.for_inference(model)
    
    # 2. Load Data & Schemas
    tables_path = getattr(inf_cfg, 'tables_path', "data/tables.json")
    schemas = load_schema_dict(tables_path) if tables_path and os.path.exists(tables_path) else {}
    if not schemas:
        logger.warning(f"No schemas loaded from {tables_path}. Inference will rely on 'instruction' field or raw question.")
    
    data_path = getattr(inf_cfg, 'data_path', None)
    if not data_path:
         # Fallback to config.paths.data_path if valid, else default
         data_path = getattr(config.paths, 'data_path', "data/test_sft.jsonl")

    if not data_path or not os.path.exists(data_path):
        raise ValueError(f"Inference data_path not found: {data_path}")
        
    logger.info(f"Loading test data from: {data_path}")
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = []
    logger.info(f"Processing {len(data)} items...")

    for item in tqdm(data, desc="Generating SQL"):
        db_id = item.get("db_id")
        question = item.get("question")
        gold = item.get("query", "N/A")

        if db_id and db_id in schemas:
            instruction = build_sft_prompt(db_id, schemas[db_id], question)
        elif "instruction" in item:
            instruction = item["instruction"]
        else:
            instruction = f"Question: {question}\nSQL:"

        system_prompt = getattr(inf_cfg, 'system_prompt', config.formatting.system_prompt)
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=getattr(inf_cfg, 'max_tokens', 128),
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = outputs[0][input_length:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Clean up markdown code blocks
        prediction = re.sub(r"```(?:sql)?\s*(.*?)\s*```", r"\1", prediction, flags=re.DOTALL | re.IGNORECASE).strip()
        prediction = " ".join(prediction.split())

        results.append({
            "db_id": db_id,
            "question": question,
            "predicted_sql": prediction,
            "gold_sql": gold,
            "is_match": prediction.lower() == gold.lower() if gold != "N/A" else None
        })

    # Prepare Output
    timestamp = datetime.now().isoformat()
    safe_model_id = model_id.replace("/", "_")
    output_dir = f"eval_results/{safe_model_id}"
    os.makedirs(output_dir, exist_ok=True)

    # If run via poe/CLI, we might want a timestamped file.
    # If run via pipeline, maybe we want a fixed name?
    # INFERENCE_EVALUATION.md says: results_{timestamp}.json
    output_filename = f"results_{timestamp.replace(':', '-').split('.')[0]}.json"
    output_path = os.path.join(output_dir, output_filename)

    final_output = {
        "metadata": {
            "model_id": model_id,
            "base_model": config.model.model_name,
            "test_dataset": data_path,
            "timestamp": timestamp
        },
        "results": results
    }

    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_id", type=str, help="Model ID or path (overrides config)")
    parser.add_argument("--data_path", type=str, help="Path to test dataset (overrides config)")
    args = parser.parse_args()
    
    app_config = load_config(args.config)

    # Update config with CLI args
    if not hasattr(app_config, 'inference'):
        app_config.inference = AppConfig({})

    if args.model_id:
        app_config.inference.model_id = args.model_id
    if args.data_path:
        app_config.inference.data_path = args.data_path

    run_inference(app_config)
