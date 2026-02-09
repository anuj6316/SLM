import os
import json
import torch
import argparse
import subprocess
from tqdm import tqdm
from unsloth import FastLanguageModel
from schema_utils import load_schema_dict, find_relevant_tables, expand_with_foreign_keys, format_schema

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
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    # 2. Load Data & Schemas
    print("Loading data and schemas...")
    schemas = load_schema_dict(tables_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Prepare Files
    gold_path = os.path.join(output_dir, "gold.txt")
    pred_path = os.path.join(output_dir, "pred.txt")
    
    print(f"Generating SQL for {len(data)} examples...")
    
    with open(gold_path, "w", encoding="utf-8") as f_gold, \
         open(pred_path, "w", encoding="utf-8") as f_pred:
        
        for ex in tqdm(data):
            db_id = ex["db_id"]
            question = ex["question"]
            gold_sql = ex["query"]
            
            # Save Gold (SQL \t db_id)
            f_gold.write(f"{gold_sql}\t{db_id}\n")
            
            # --- Inference Logic (Same as Training) ---
            schema = schemas[db_id]
            matched = find_relevant_tables(question, schema)
            selected_tables = expand_with_foreign_keys(matched, schema)
            schema_text = format_schema(db_id, schema, selected_tables)
            
            prompt = f"""<|im_start|>system
You are a Text-to-SQL expert.<|im_end|>
<|im_start|>user
{schema_text}

Question: {question}
SQL:<|im_end|>
<|im_start|>assistant
"""
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False, # Greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode and clean
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract SQL (after "assistant\n")
            # Note: unsloth decoding might include the prompt or not depending on config.
            # Usually safe to split by assistant tag if present, or just take the end.
            # With format_example in training, we expect the output to follow the prompt.
            
            # Reliable extraction based on prompt structure
            if "SQL:" in generated_text:
                # Fallback if the template rendering behaves differently
                pred_sql = generated_text.split("SQL:")[-1].strip()
            else:
                 pred_sql = generated_text.strip()
            
            # Specifically handle the chat template output if it repeats input
            if "assistant\n" in pred_sql:
                pred_sql = pred_sql.split("assistant\n")[-1].strip()
                
            # Remove any trailing markdown like ```sql ... ```
            pred_sql = pred_sql.replace("```sql", "").replace("```", "").strip()
            
            # One line
            pred_sql = " ".join(pred_sql.split())
            
            f_pred.write(f"{pred_sql}\n")

    # 4. Run Evaluation
    print("\nRunning Official Spider Evaluation...")
    cmd = [
        "python3", spider_eval_path,
        "--gold", gold_path,
        "--pred", pred_path,
        "--db", db_dir,
        "--table", tables_path,
        "--etype", "all"  # exec + match
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Evaluation failed!")
        print(e)

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
