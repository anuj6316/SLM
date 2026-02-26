"""
CLI Commands - Individual command handlers for the SLM pipeline.

Commands:
    cmd_data: Handle data pipeline commands
    cmd_train: Handle training commands
    cmd_eval: Handle evaluation commands
    cmd_infer: Handle inference commands
    cmd_pipeline: Handle full pipeline commands
"""

import logging
from pathlib import Path

from slm.config import settings
from slm.data import DataPipeline
from slm.training import SFTTrainer
from slm.evaluation import Evaluator
from slm.inference import InferenceEngine

logger = logging.getLogger(__name__)


def cmd_data(args) -> None:
    """Handle data commands."""
    pipeline = DataPipeline(settings)

    if args.action == "gather":
        logger.info("=== Gathering datasets ===")
        pipeline.gather()
    elif args.action == "process":
        logger.info("=== Processing datasets ===")
        pipeline.process()
    elif args.action == "split":
        logger.info("=== Splitting datasets ===")
        pipeline.split()
    elif args.action == "all":
        logger.info("=== Running full data pipeline ===")
        pipeline.run_all()
    else:
        logger.error(f"Unknown action: {args.action}")


def cmd_train(args) -> None:
    """Handle training commands."""
    trainer = SFTTrainer(settings)

    if args.action == "sft":
        logger.info("=== Starting SFT Training ===")
        output_dir = trainer.train()
        logger.info(f"Model saved to: {output_dir}")
    elif args.action == "grpo":
        logger.error("GRPO training not yet implemented")
    else:
        logger.error(f"Unknown action: {args.action}")


def cmd_eval(args) -> None:
    """Handle evaluation commands."""
    model_path = Path(args.model_path) if args.model_path else None
    evaluator = Evaluator(settings, model_path)

    data_path = Path(args.data_path) if args.data_path else settings.data.val_file
    output_path = Path(args.output_path) if args.output_path else None

    logger.info(f"=== Evaluating on {data_path} ===")
    results = evaluator.evaluate_dataset(data_path, output_path)

    if "metrics" in results:
        logger.info(f"Accuracy: {results['metrics']['exact_match_accuracy']:.2%}")


def cmd_infer(args) -> None:
    """Handle inference commands."""
    model_path = Path(args.model_path) if args.model_path else None
    engine = InferenceEngine(settings, model_path)

    if args.interactive:
        logger.info("=== Interactive Mode ===")
        engine.setup()
        engine.load_schemas()

        print("\nText-to-SQL Inference (type 'quit' to exit)")
        print("-" * 40)

        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break

                db_id = input("DB ID (default: concert_singer): ").strip()
                db_id = db_id or "concert_singer"

                result = engine.generate(question, db_id)
                print(f"\nSQL: {result['sql']}")
                print(f"Tokens: {result['tokens_used']}")
            except KeyboardInterrupt:
                break

        print("\nGoodbye!")

    else:
        data_path = (
            Path(args.data_path) if args.data_path else settings.inference.data_path
        )
        output_path = (
            Path(args.output_path)
            if args.output_path
            else settings.inference.output_path
        )
        num_samples = args.num_samples or settings.inference.num_samples

        use_tracing = args.trace
        if use_tracing:
            logger.info("=== Running traced inference ===")
        else:
            logger.info("=== Running inference ===")

        results = engine.run_batch(
            data_path=data_path,
            num_samples=num_samples,
            output_path=output_path,
            use_tracing=use_tracing,
        )

        exact_matches = sum(1 for r in results if r.get("exact_match"))
        accuracy = exact_matches / len(results) if results else 0
        logger.info(f"Results: {len(results)} samples, {accuracy:.2%} accuracy")


def cmd_pipeline(args) -> None:
    """Handle full pipeline commands."""
    if args.action == "full":
        logger.info("=== Running Full Pipeline ===")

        logger.info("\n[1/4] Data Pipeline")
        data_pipeline = DataPipeline(settings)
        data_pipeline.run_all()

        logger.info("\n[2/4] Training")
        trainer = SFTTrainer(settings)
        trainer.train()

        logger.info("\n[3/4] Evaluation")
        evaluator = Evaluator(settings)
        evaluator.evaluate_dataset(settings.data.val_file)

        logger.info("\n[4/4] Inference")
        engine = InferenceEngine(settings)
        engine.run_batch(
            data_path=settings.data.val_file,
            num_samples=100,
            output_path=settings.evaluation.output_dir / "inference_results.json",
        )

        logger.info("\n=== Pipeline Complete ===")

    elif args.action == "sft-eval":
        logger.info("=== Running SFT + Eval Pipeline ===")

        logger.info("\n[1/2] Training")
        trainer = SFTTrainer(settings)
        trainer.train()

        logger.info("\n[2/2] Evaluation")
        evaluator = Evaluator(settings)
        evaluator.evaluate_dataset(settings.data.val_file)

        logger.info("\n=== Pipeline Complete ===")

    else:
        logger.error(f"Unknown action: {args.action}")
