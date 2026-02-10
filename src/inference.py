import os
import json
import torch
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel
from schema_utils import load_schema_dict, build_sft_prompt
from utils import get_logger

logger = get_logger(__name__)

def run_inference(
    model_path,
    data_path,
    tables_path,
    output_path,
    system_prompt="You are a Text-to-SQL expert.",
    max_new_tokens=128
):
    # 1. Load Model
    logger.info(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    # 2. Load Data & Schemas
    logger.info(f"Loading schemas from {tables_path}")
    schemas = load_schema_dict(tables_path)

    logger.info(f"Loading data from: {data_path}")
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = []
    logger.info(f"Running inference on {len(data)} items...")

    for item in tqdm(data, desc="Inference"):
        db_id = item.get("db_id")
        # If it's a pre-built JSONL, it has 'instruction'
        # If it's raw Spider, it has 'question'
        question = item.get("question") or item.get("instruction")
        gold = item.get("query") or item.get("output", "N/A")

        if not db_id or not question:
            logger.warning(f"Skipping item due to missing db_id or question: {item}")
            continue

        schema = schemas.get(db_id)
        if not schema:
            logger.error(f"Schema not found for db_id: {db_id}")
            continue

        # CRITICAL FIX: Build prompt with schema injection
        instruction = build_sft_prompt(db_id, schema, question)
        
        prompt = (
<<<<<<< HEAD
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
=======
            f"""<|im_start|>system
{system_prompt}<|im_end|>
"""
            f"""<|im_start|>user
{instruction}<|im_end|>
"""
            f"""<|im_start|>assistant
"""
>>>>>>> d4798f6 (fixed all the syntax errors in inference)
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
<<<<<<< HEAD
=======
        # Clean up: extract only the assistant's response
>>>>>>> d4798f6 (fixed all the syntax errors in inference)
        if "assistant\n" in full_output:
            prediction = full_output.split("assistant\n")[-1].strip()
        else:
            prediction = full_output.split("SQL:")[-1].strip() if "SQL:" in full_output else full_output.strip()

        prediction = prediction.replace("```sql", "").replace("```", "").strip()

        results.append({
            "db_id": db_id,
            "question": question,
            "gold_query": gold,
            "model_output": prediction
        })

    # 3. Save Results
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tables_path", type=str, default="data/spider/tables.json")
    parser.add_argument("--output_path", type=str, default="inference_results.json")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--max_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        tables_path=args.tables_path,
        output_path=args.output_path,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_results.json")
    parser.add_argument("--system_prompt", type=str, default="You are a Text-to-SQL expert.")
    parser.add_argument("--max_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens
    )
