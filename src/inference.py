import os
import json
import torch
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel

def run_inference(
    model_path,
    data_path,
    output_path,
    system_prompt="You are a Text-to-SQL expert.",
    max_new_tokens=128
):
    # 1. Load Model
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    # 2. Load Data
    print(f"Loading data from: {data_path}")
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = []
    print(f"Running inference on {len(data)} items...")

    for item in tqdm(data):
        # Handle different field names for flexibility
        # 'instruction' is from our SFT JSONL, 'question' is from raw Spider
        instruction = item.get("instruction") or item.get("question", "")
        # 'output' is from our SFT JSONL, 'query' is from raw Spider
        gold = item.get("output") or item.get("query", "N/A")
        
        prompt = (
            f"<|im_start|>system
{system_prompt}<|im_end|>
"
            f"<|im_start|>user
{instruction}<|im_end|>
"
            f"<|im_start|>assistant
"
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False, # Deterministic output
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Clean up: extract only the assistant's response
        if "assistant
" in full_output:
            prediction = full_output.split("assistant
")[-1].strip()
        else:
            # Fallback if splitting fails
            prediction = full_output.strip()

        # Remove potential markdown
        prediction = prediction.replace("```sql", "").replace("```", "").strip()

        results.append({
            "input": instruction,
            "gold_query": gold,
            "model_output": prediction
        })

    # 3. Save Results
    print(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

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
