"""
Evaluator - Evaluate Text-to-SQL models on benchmarks.

Usage:
    from slm.evaluation import Evaluator
    from slm.config import settings

    evaluator = Evaluator(settings)
    results = evaluator.evaluate_dataset("data/val_split.jsonl")
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from slm.config import Settings
from slm.inference import InferenceEngine
from slm.evaluation.metrics import calculate_exact_match, extract_question

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate Text-to-SQL models on standard benchmarks.

    Supports:
        - Spider benchmark (exact match + execution)
        - Custom JSONL datasets
    """

    def __init__(self, settings: Settings, model_path: Optional[Path] = None) -> None:
        self._settings = settings
        self._model_path = model_path or Path(settings.training.output_dir)
        self._engine = InferenceEngine(settings, model_path)
        self._results_dir = settings.evaluation.output_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(
        self,
        data_path: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a JSONL dataset.

        Args:
            data_path: Path to JSONL dataset
            output_path: Optional path to save results

        Returns:
            Dict with accuracy metrics and per-sample results
        """
        if self._engine.model is None:
            self._engine.setup()
        if self._engine.schema_map is None:
            self._engine.load_schemas()

        data = []
        with open(data_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)

        predictions = []
        gold_queries = []
        db_ids = []

        for entry in tqdm(data, desc="Evaluating"):
            question = extract_question(entry)
            db_id = entry.get("metadata", {}).get("db_id", "unknown")
            gold_sql = entry.get("output", "")

            result = self._engine.generate(question, db_id)

            predictions.append(result["sql"])
            gold_queries.append(gold_sql)
            db_ids.append(db_id)

        exact_matches, accuracy = calculate_exact_match(predictions, gold_queries)

        results = {
            "metadata": {
                "model": str(self._model_path),
                "dataset": str(data_path),
                "num_samples": len(predictions),
                "timestamp": datetime.now().isoformat(),
            },
            "metrics": {
                "exact_match_accuracy": accuracy,
                "exact_matches": exact_matches,
                "total_samples": len(predictions),
            },
            "per_sample": [
                {
                    "db_id": db_ids[i],
                    "prediction": predictions[i],
                    "gold": gold_queries[i],
                    "exact_match": predictions[i].lower().strip()
                    == gold_queries[i].lower().strip(),
                }
                for i in range(len(predictions))
            ],
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        if self._settings.mlflow.enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._settings.mlflow.tracking_uri)
                mlflow.set_experiment(self._settings.mlflow.experiment_name)
                
                with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_params({
                        "dataset": str(data_path),
                        "model": str(self._model_path),
                        "num_samples": len(predictions),
                    })
                    mlflow.log_metrics({
                        "exact_match_accuracy": accuracy,
                        "exact_matches": float(exact_matches),
                        "total_samples": float(len(predictions)),
                    })
                    logger.info("Evaluation metrics logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        logger.info(
            f"Exact Match Accuracy: {accuracy:.2%} ({exact_matches}/{len(predictions)})"
        )
        return results

    def evaluate_spider(
        self,
        dev_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        spider_eval_script: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on Spider benchmark using official evaluation script.

        Args:
            dev_path: Path to Spider dev.json
            db_path: Path to Spider databases directory
            spider_eval_script: Path to evaluation script

        Returns:
            Dict with accuracy metrics
        """
        dev_path = dev_path or self._settings.data.raw_dir / "spider_dev.json"
        db_path = db_path or self._settings.data.raw_dir / "databases"

        if not dev_path.exists():
            logger.error(f"Spider dev set not found: {dev_path}")
            return {}

        if self._engine.model is None:
            self._engine.setup()
        if self._engine.schema_map is None:
            self._engine.load_schemas()

        with open(dev_path, "r") as f:
            dev_data = json.load(f)

        predictions = []
        gold_queries = []

        for entry in tqdm(dev_data, desc="Evaluating Spider"):
            question = entry.get("question", "")
            db_id = entry.get("db_id", "unknown")
            gold_sql = entry.get("query", "")

            result = self._engine.generate(question, db_id)

            predictions.append(result["sql"])
            gold_queries.append(gold_sql)

        pred_file = self._results_dir / "predictions.txt"
        gold_file = self._results_dir / "gold.txt"

        with open(pred_file, "w") as f:
            f.write("\n".join(predictions))
        with open(gold_file, "w") as f:
            f.write("\n".join(gold_queries))

        exact_matches, exact_match_accuracy = calculate_exact_match(
            predictions, gold_queries
        )

        results = {
            "exact_match_accuracy": exact_match_accuracy,
            "exact_matches": exact_matches,
            "total_samples": len(predictions),
        }

        if spider_eval_script and db_path and spider_eval_script.exists():
            try:
                result = subprocess.run(
                    [
                        "python",
                        str(spider_eval_script),
                        "--gold",
                        str(gold_file),
                        "--pred",
                        str(pred_file),
                        "--db",
                        str(db_path),
                        "--etype",
                        "all",
                    ],
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Spider evaluation:\n{result.stdout}")
            except Exception as e:
                logger.error(f"Failed to run Spider evaluation: {e}")

        logger.info(f"Spider Exact Match: {exact_match_accuracy:.2%}")
        return results

    def compare_models(
        self,
        model_paths: List[Path],
        data_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.

        Args:
            model_paths: List of paths to models
            data_path: Path to evaluation dataset

        Returns:
            Dict mapping model path to evaluation results
        """
        results = {}

        for model_path in model_paths:
            logger.info(f"\n=== Evaluating {model_path} ===")
            evaluator = Evaluator(self._settings, model_path)
            output_path = self._results_dir / f"{model_path.name}_eval.json"
            results[str(model_path)] = evaluator.evaluate_dataset(
                data_path, output_path
            )

        comparison = {
            "metadata": {
                "dataset": str(data_path),
                "models_compared": [str(p) for p in model_paths],
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
        }

        comparison_path = self._results_dir / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        return results
