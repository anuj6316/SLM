import argparse
import subprocess
import sys
import os

# Default Paths
SPIDER_DIR = "data/spider"
DATA_OUTPUT = "data/spider_sft/train_sft.jsonl"
MODEL_OUTPUT = "outputs/qwen-text2sql"
EVAL_OUTPUT = "evaluation_results"

def run_command(command, description):
    print(f"\n[PIPELINE] Starting: {description}...")
    print(f"Command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"[PIPELINE] Success: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"[PIPELINE] Failed: {description}")
        sys.exit(e.returncode)

def run_prepare(args):
    cmd = [
        sys.executable, "src/build_jsonl_sft.py",
        "--spider_dir", args.spider_dir,
        "--output_path", args.data_output
    ]
    run_command(cmd, "Data Preparation")

def run_train(args):
    cmd = [
        sys.executable, "src/train.py",
        "--data_path", args.data_output,
        "--output_dir", args.model_output,
        "--epochs", str(args.epochs),
        "--system_prompt", args.system_prompt
    ]
    if args.base_model:
        cmd.extend(["--model_name", args.base_model])
        
    run_command(cmd, "Model Training")

def run_evaluate(args):
    # Ensure model exists before evaluating
    if not os.path.exists(args.model_output):
        print(f"Error: Model not found at {args.model_output}. Train first?")
        sys.exit(1)

    cmd = [
        sys.executable, "src/evaluation_pipeline.py",
        "--model_path", args.model_output,
        "--data_path", os.path.join(args.spider_dir, "dev.json"),
        "--tables_path", os.path.join(args.spider_dir, "tables.json"),
        "--db_dir", os.path.join(args.spider_dir, "database"),
        "--output_dir", args.eval_output
    ]
    run_command(cmd, "Model Evaluation")

def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline Orchestrator")
    
    # Global arguments
    parser.add_argument("--spider_dir", default=SPIDER_DIR, help="Path to Spider dataset root")
    parser.add_argument("--data_output", default=DATA_OUTPUT, help="Path for SFT jsonl file")
    parser.add_argument("--model_output", default=MODEL_OUTPUT, help="Path to save/load trained model")
    parser.add_argument("--eval_output", default=EVAL_OUTPUT, help="Path to save evaluation results")
    parser.add_argument("--system_prompt", default="You are a Text-to-SQL expert.", help="System prompt for the chat template")
    
    subparsers = parser.add_subparsers(dest="stage", help="Pipeline Stage")
    
    # Stage: Prepare
    parser_prepare = subparsers.add_parser("prepare", help="Prepare SFT dataset")
    
    # Stage: Train
    parser_train = subparsers.add_parser("train", help="Fine-tune the model")
    parser_train.add_argument("--epochs", type=int, default=2)
    parser_train.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    
    # Stage: Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate the model on Dev set")
    
    # Stage: All
    parser_all = subparsers.add_parser("all", help="Run complete pipeline")
    parser_all.add_argument("--epochs", type=int, default=2)
    parser_all.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    args = parser.parse_args()

    if args.stage == "prepare":
        run_prepare(args)
    elif args.stage == "train":
        run_train(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    elif args.stage == "all":
        run_prepare(args)
        run_train(args)
        run_evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()