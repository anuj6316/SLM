import os
import json
import torch
import argparse
import subprocess
from tqdm import tqdm
from unsloth import FastLanguageModel
from src.schema_utils import load_schema_dict, build_sft_prompt
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_model(
    model_path,
    data_path,
    tables_path,
    db_dir,
    output_dir,
    spider_eval_path="data/spider/spider-master/evaluation.py"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    logger.info(f"Loading model for evaluation: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    # 2. Load Data & Schemas
    logger.info("Loading schemas and validation data...")
    schemas = load_schema_dict(tables_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Prepare Files
    gold_path = os.path.join(output_dir, "gold.txt")
    pred_path = os.path.join(output_dir, "pred.txt")
    
    logger.info(f"Generating SQL for {len(data)} examples...")
    
    with open(gold_path, "w", encoding="utf-8") as f_gold, \
         open(pred_path, "w", encoding="utf-8") as f_pred:
        
        for ex in tqdm(data, desc="Evaluating"):
            db_id = ex["db_id"]
            question = ex["question"]
            gold_sql = ex["query"]
            
            # Save Gold (SQL \t db_id)
            f_gold.write(f"{gold_sql}\t{db_id}\n")
            
            # --- Inference Logic (Consistent with Training) ---
            schema = schemas[db_id]
            # Use centralized prompt builder
            instruction = build_sft_prompt(db_id, schema, question)
            
            prompt = (
                f"<|im_start|>system\nYou are a Text-to-SQL expert.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract SQL after assistant tag
            if "assistant\n" in generated_text:
                pred_sql = generated_text.split("assistant\n")[-1].strip()
            else:
                # If chat template wasn't applied or returned whole prompt
                pred_sql = generated_text.split("SQL:")[-1].strip() if "SQL:" in generated_text else generated_text.strip()
                
            # Clean up
            pred_sql = pred_sql.replace("```sql", "").replace("```", "").strip()
            pred_sql = " ".join(pred_sql.split())
            
            f_pred.write(f"{pred_sql}\n")

    # 4. Run Evaluation
    logger.info("Running Official Spider Evaluation script...")
    cmd = [
        "python3", spider_eval_path,
        "--gold", gold_path,
        "--pred", pred_path,
        "--db", db_dir,
        "--table", tables_path,
        "--etype", "all"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation script failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/spider/dev.json")
    parser.add_argument("--tables_path", type=str, default="data/spider/tables.json")
    parser.add_argument("--db_dir", type=str, default="data/spider/database")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        tables_path=args.tables_path,
        db_dir=args.db_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/spider/dev.json")
    parser.add_argument("--tables_path", type=str, default="data/spider/tables.json")
    parser.add_argument("--db_dir", type=str, default="data/spider/database")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        tables_path=args.tables_path,
        db_dir=args.db_dir,
        output_dir=args.output_dir
    )
