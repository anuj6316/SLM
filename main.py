import argparse
import subprocess
import sys
import os
import logging
from typing import List, Optional

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Orchestrator")

# Default Constants
DEFAULTS = {
    "SPIDER_DIR": "data/spider",
    "DATA_OUTPUT": "data/spider_sft/train_sft.jsonl",
    "MODEL_OUTPUT": "outputs/qwen-text2sql",
    "EVAL_OUTPUT": "evaluation_results",
    "BASE_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
    "SYSTEM_PROMPT": "You are a Text-to-SQL expert."
}

def run_command(command: List[str], description: str):
    """
    Executes a shell command and streams its output to the logger.
    """
    logger.info(f"--- Starting: {description} ---")
    logger.debug(f"Command: {' '.join(command)}")
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            bufsize=1
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end="") # Print to console directly to preserve formatting from sub-scripts
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage '{description}' failed with exit code {process.returncode}.")
            sys.exit(process.returncode)
            
        logger.info(f"--- Completed: {description} ---\n")
        
    except Exception as e:
        logger.critical(f"Failed to execute {description}: {str(e)}")
        sys.exit(1)

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
    if not os.path.exists(args.model_output):
        logger.error(f"Model not found at {args.model_output}. Please train the model first.")
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

def run_inference(args):
    if not os.path.exists(args.model_output):
        logger.error(f"Model not found at {args.model_output}.")
        sys.exit(1)

    cmd = [
        sys.executable, "src/inference.py",
        "--model_path", args.model_output,
        "--data_path", args.inference_data,
        "--output_path", args.inference_output,
        "--system_prompt", args.system_prompt
    ]
    run_command(cmd, "Model Inference")

def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline Orchestrator")

    # Global arguments
    parser.add_argument("--spider_dir", default=DEFAULTS["SPIDER_DIR"], help="Path to Spider dataset root")
    parser.add_argument("--data_output", default=DEFAULTS["DATA_OUTPUT"], help="Path for SFT jsonl file")
    parser.add_argument("--model_output", default=DEFAULTS["MODEL_OUTPUT"], help="Path to save/load trained model")
    parser.add_argument("--eval_output", default=DEFAULTS["EVAL_OUTPUT"], help="Path to save evaluation results")
    parser.add_argument("--system_prompt", default=DEFAULTS["SYSTEM_PROMPT"], help="System prompt for the chat template")

    subparsers = parser.add_subparsers(dest="stage", help="Pipeline Stage")

    # Stage: Prepare
    subparsers.add_parser("prepare", help="Prepare SFT dataset")

    _extracted_from_main_17(subparsers, "train", "Fine-tune the model")
    # Stage: Evaluate
    subparsers.add_parser("evaluate", help="Evaluate the model on Dev set")

    # Stage: Inference
    parser_inf = subparsers.add_parser("inference", help="Run batch inference")
    parser_inf.add_argument("--inference_data", required=True, help="Path to input JSON/JSONL file")
    parser_inf.add_argument("--inference_output", default="results.json", help="Path to save JSON results")

    _extracted_from_main_17(subparsers, "all", "Run complete pipeline")
    args = parser.parse_args()

    if args.stage == "prepare":
        run_prepare(args)
    elif args.stage == "train":
        run_train(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    elif args.stage == "inference":
        run_inference(args)
    elif args.stage == "all":
        run_prepare(args)
        run_train(args)
        run_evaluate(args)
    else:
        parser.print_help()


# TODO Rename this here and in `main`
def _extracted_from_main_17(subparsers, arg1, help):
    # Stage: Train
    parser_train = subparsers.add_parser(arg1, help=help)
    parser_train.add_argument("--epochs", type=int, default=2)
    parser_train.add_argument("--base_model", type=str, default=DEFAULTS["BASE_MODEL"])

if __name__ == "__main__":
    main()