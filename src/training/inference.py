import os
import json
import torch
import argparse
import logging
import re
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
    model_path = getattr(inf_cfg, 'model_path', config.paths.model_output)
    logger.info(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    
    # 2. Load Data & Schemas
    tables_path = getattr(inf_cfg, 'tables_path', None)
    schemas = load_schema_dict(tables_path) if tables_path else {}
    
    data_path = getattr(inf_cfg, 'data_path', None)
    if not data_path:
        raise ValueError("Inference data_path must be specified in config or CLI.")
        
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
        prediction = re.sub(r"```(?:sql)?\s*(.*?)\s*```", r"\1", prediction, flags=re.DOTALL | re.IGNORECASE).strip()
        prediction = " ".join(prediction.split())

        results.append({
            "db_id": db_id,
            "question": question,
            "gold_query": gold,
            "model_output": prediction
        })

    output_path = getattr(inf_cfg, 'output_path', "inference_results.json")
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    app_config = load_config(args.config)
    run_inference(app_config)