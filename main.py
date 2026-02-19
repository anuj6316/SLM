import argparse
import sys
import os
import logging
from typing import List, Optional

# Add src to path to allow direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from core.config import load_config, AppConfig
from preprocessing.pipeline import run_pipeline
from training.train import train_model
from training.evaluate import evaluate_model
from training.inference import run_inference as run_model_inference

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log")
    ]
)
logger = logging.getLogger("Orchestrator")

def execute_stage(stage_name: str, func, config: AppConfig):
    """Generic executor for pipeline stages."""
    logger.info(f"--- Starting Stage: {stage_name} ---")
    try:
        func(config)
        logger.info(f"--- Stage {stage_name} Completed Successfully ---\n")
    except Exception as e:
        logger.error(f"Stage {stage_name} Failed: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Pipeline Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("stage", choices=["preprocess", "train", "evaluate", "inference", "all"], 
                        help="Pipeline Stage to execute")
    
    # Optional CLI overrides (for quick testing)
    parser.add_argument("--data_path", help="Override data path")
    parser.add_argument("--epochs", type=int, help="Override training epochs")

    args = parser.parse_args()
    
    # 1. Load Single Source of Truth
    config = load_config(args.config)
    
    # 2. Apply dynamic overrides from CLI if provided
    if args.data_path:
        config.paths.data_path = args.data_path
    if args.epochs:
        config.training.epochs = args.epochs

    # 3. Execution Logic
    if args.stage == "preprocess":
        execute_stage("Preprocessing", run_pipeline, config)
    elif args.stage == "train":
        execute_stage("Training", train_model, config)
    elif args.stage == "evaluate":
        execute_stage("Evaluation", evaluate_model, config)
    elif args.stage == "inference":
        execute_stage("Inference", run_model_inference, config)
    elif args.stage == "all":
        execute_stage("Preprocessing", run_pipeline, config)
        execute_stage("Training", train_model, config)
        execute_stage("Evaluation", evaluate_model, config)

if __name__ == "__main__":
    main()
