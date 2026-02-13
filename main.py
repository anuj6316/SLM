import argparse
import subprocess
import sys
import os
import logging
from typing import List, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from config_utils import load_config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Orchestrator")

# Default Constants
DEFAULTS = {
    "DATA_PATH": "data/train_sft.jsonl",
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

def run_preprocess(args):
    cmd = [
        sys.executable, "src/preprocess.py",
        "--config", args.config
    ]
    run_command(cmd, "Data Preprocessing")

def run_train(args):
    # Load config for additional parameters
    config = load_config(args.config)
    train_config = config.get("training", {})
    
    cmd = [
        sys.executable, "src/train.py",
        "--config", args.config, # Pass config to sub-script
        "--data_path", args.data_path,
        "--output_dir", args.model_output,
        "--epochs", str(args.epochs),
        "--system_prompt", args.system_prompt
    ]
    if args.base_model:
        cmd.extend(["--model_name", args.base_model])
    elif "model_name" in train_config:
        cmd.extend(["--model_name", train_config["model_name"]])
        
    run_command(cmd, "Model Training")

def run_evaluate(args):
    if not os.path.exists(args.model_output):
        logger.error(f"Model not found at {args.model_output}. Please train the model first.")
        sys.exit(1)

    cmd = [
        sys.executable, "src/evaluation_pipeline.py",
        "--config", args.config, # Pass config to sub-script
        "--model_path", args.model_output,
        "--data_path", args.data_path,
        "--output_dir", args.eval_output
    ]
    run_command(cmd, "Model Evaluation")

def run_inference(args):
    if not os.path.exists(args.model_output):
        logger.error(f"Model not found at {args.model_output}.")
        sys.exit(1)

    cmd = [
        sys.executable, "src/inference.py",
        "--config", args.config, # Pass config to sub-script
        "--model_path", args.model_output,
        "--data_path", args.inference_data,
        "--output_path", args.inference_output,
        "--system_prompt", args.system_prompt
    ]
    run_command(cmd, "Model Inference")

def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")

    # Shared parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--data_path", default=DEFAULTS["DATA_PATH"], help="Path to SFT jsonl file")
    parent_parser.add_argument("--model_output", default=DEFAULTS["MODEL_OUTPUT"], help="Path to save/load trained model")
    parent_parser.add_argument("--eval_output", default=DEFAULTS["EVAL_OUTPUT"], help="Path to save evaluation results")
    parent_parser.add_argument("--system_prompt", default=DEFAULTS["SYSTEM_PROMPT"], help="System prompt for the chat template")

    subparsers = parser.add_subparsers(dest="stage", help="Pipeline Stage")

    # Stage: Preprocess
    subparsers.add_parser("preprocess", help="Run data loading and preprocessing", parents=[parent_parser])

    # Stage: Train
    parser_train = subparsers.add_parser("train", help="Fine-tune the model", parents=[parent_parser])
    parser_train.add_argument("--epochs", type=int, default=2)
    parser_train.add_argument("--base_model", type=str, default=DEFAULTS["BASE_MODEL"])

    # Stage: Evaluate
    subparsers.add_parser("evaluate", help="Evaluate the model", parents=[parent_parser])

    # Stage: Inference
    parser_inf = subparsers.add_parser("inference", help="Run batch inference", parents=[parent_parser])
    parser_inf.add_argument("--inference_data", required=True, help="Path to input JSON/JSONL file")
    parser_inf.add_argument("--inference_output", default="results.json", help="Path to save JSON results")

    # Stage: All
    parser_all = subparsers.add_parser("all", help="Run complete pipeline", parents=[parent_parser])
    parser_all.add_argument("--epochs", type=int, default=2)
    parser_all.add_argument("--base_model", type=str, default=DEFAULTS["BASE_MODEL"])

    args = parser.parse_args()
    
    # Load config and override defaults if not provided in CLI
    config = load_config(args.config)
    
    # Update args with config values if args are still at their default values
    # This is a bit tricky with argparse, but we can check if the value in config exists
    # and if the user didn't provide a value on CLI.
    # For simplicity, we'll merge them in the run_* functions.
    
    if args.stage == "preprocess":
        run_preprocess(args)
    elif args.stage == "train":
        run_train(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    elif args.stage == "inference":
        run_inference(args)
    elif args.stage == "all":
        run_preprocess(args)
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