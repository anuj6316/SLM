"""
CLI Module - Unified command-line interface for SLM Text-to-SQL pipeline.

Usage:
    python main.py data gather    # Download datasets
    python main.py data process   # Clean + format datasets
    python main.py data split     # Train/val split
    python main.py data all       # Full data pipeline

    python main.py train sft      # Run SFT training

    python main.py eval           # Evaluate on dataset

    python main.py infer          # Run inference
    python main.py infer --trace  # Run with MLflow tracing

    python main.py pipeline full  # Run complete pipeline
"""

import argparse
import logging

from slm.cli.commands import cmd_data, cmd_train, cmd_eval, cmd_infer, cmd_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SLM Text-to-SQL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    _setup_data_parser(subparsers)
    _setup_train_parser(subparsers)
    _setup_eval_parser(subparsers)
    _setup_infer_parser(subparsers)
    _setup_pipeline_parser(subparsers)

    args = parser.parse_args()

    if args.command == "data":
        cmd_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)


def _setup_data_parser(subparsers) -> None:
    """Setup data subparser."""
    data_parser = subparsers.add_parser("data", help="Data pipeline")
    data_parser.add_argument(
        "action",
        choices=["gather", "process", "split", "all"],
        help="Action to perform",
    )


def _setup_train_parser(subparsers) -> None:
    """Setup train subparser."""
    train_parser = subparsers.add_parser("train", help="Training pipeline")
    train_parser.add_argument("action", choices=["sft", "grpo"], help="Training method")


def _setup_eval_parser(subparsers) -> None:
    """Setup eval subparser."""
    eval_parser = subparsers.add_parser("eval", help="Evaluation pipeline")
    eval_parser.add_argument("--data-path", help="Path to evaluation dataset")
    eval_parser.add_argument(
        "--model-path", help="Path to model (default: from config)"
    )
    eval_parser.add_argument("--output-path", help="Path to save results")


def _setup_infer_parser(subparsers) -> None:
    """Setup infer subparser."""
    infer_parser = subparsers.add_parser("infer", help="Inference pipeline")
    infer_parser.add_argument("--data-path", help="Path to input dataset")
    infer_parser.add_argument("--model-path", help="Path to model")
    infer_parser.add_argument("--output-path", help="Path to save results")
    infer_parser.add_argument("--num-samples", type=int, help="Number of samples")
    infer_parser.add_argument(
        "--trace", action="store_true", help="Enable MLflow tracing"
    )
    infer_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )


def _setup_pipeline_parser(subparsers) -> None:
    """Setup pipeline subparser."""
    pipeline_parser = subparsers.add_parser("pipeline", help="Full pipeline")
    pipeline_parser.add_argument(
        "action", choices=["full", "sft-eval"], help="Pipeline to run"
    )


if __name__ == "__main__":
    main()
