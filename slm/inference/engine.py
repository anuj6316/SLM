"""
Inference Engine - Local and MLflow-traced inference for Text-to-SQL.

Usage:
    from slm.inference import InferenceEngine
    from slm.config import settings

    engine = InferenceEngine(settings)
    result = engine.generate("How many singers?", "concert_singer")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from slm.config import Settings
from slm.utils import load_schema_dict, clean_sql, extract_question_from_input
from slm.inference.tracing import setup_mlflow, traced_generate

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Text-to-SQL inference engine with optional MLflow tracing."""

    def __init__(self, settings: Settings, model_path: Optional[Path] = None) -> None:
        self._settings = settings
        self._model_path = model_path or Path(settings.training.output_dir)
        self._model = None
        self._tokenizer = None
        self._schema_map: Optional[Dict[str, str]] = None
        self._mlflow_enabled = False

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def schema_map(self) -> Optional[Dict[str, str]]:
        return self._schema_map

    def setup(self) -> None:
        """Load model and tokenizer."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")

        logger.info(f"Loading model from: {self._model_path}")

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self._model_path),
            max_seq_length=self._settings.model.max_seq_length,
            dtype=self._settings.model.dtype,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self._model)
        logger.info("Model loaded successfully")

    def load_schemas(self, tables_path: Optional[Path] = None) -> None:
        """Load database schemas."""
        tables_path = tables_path or self._settings.data.tables_file
        self._schema_map = load_schema_dict(tables_path)

    def setup_mlflow(self) -> None:
        """Configure MLflow for Databricks tracing."""
        if not self._settings.mlflow.enabled:
            return

        self._mlflow_enabled = setup_mlflow(
            tracking_uri=self._settings.mlflow.tracking_uri,
            experiment_name=self._settings.mlflow.experiment_name,
        )

    def _build_prompt(self, question: str, db_id: str) -> str:
        """Build the full prompt for inference."""
        if self._schema_map and db_id in self._schema_map:
            schema = self._schema_map[db_id]
        else:
            schema = f"Schema for {db_id} not available"

        return f"""<|im_start|>system
{self._settings.formatting.system_prompt}<|im_end|>
<|im_start|>user
### Database Schema:
{schema}

### Question:
{question}<|im_end|>
<|im_start|>assistant
"""

    def generate(self, question: str, db_id: str = "default") -> Dict[str, Any]:
        """
        Generate SQL from a natural language question.

        Args:
            question: Natural language question
            db_id: Database identifier

        Returns:
            Dict with question, db_id, sql, tokens_used
        """
        if self._model is None:
            self.setup()

        if self._schema_map is None:
            self.load_schemas()

        prompt = self._build_prompt(question, db_id)

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._settings.inference.max_new_tokens,
                use_cache=True,
                do_sample=self._settings.inference.do_sample,
                temperature=self._settings.inference.temperature
                if self._settings.inference.do_sample
                else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        sql = clean_sql(generated_text)

        return {
            "question": question,
            "db_id": db_id,
            "sql": sql,
            "raw_output": generated_text,
            "input_tokens": input_length,
            "output_tokens": len(new_tokens),
            "tokens_used": input_length + len(new_tokens),
        }

    def generate_traced(self, question: str, db_id: str = "default") -> Dict[str, Any]:
        """
        Generate SQL with MLflow tracing.

        Args:
            question: Natural language question
            db_id: Database identifier

        Returns:
            Dict with question, db_id, sql, tokens_used
        """
        return traced_generate(
            generate_fn=self.generate,
            question=question,
            db_id=db_id,
            mlflow_enabled=self._mlflow_enabled,
        )

    def run_batch(
        self,
        data_path: Optional[Path] = None,
        num_samples: Optional[int] = None,
        output_path: Optional[Path] = None,
        use_tracing: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a dataset.

        Args:
            data_path: Path to JSONL dataset
            num_samples: Max samples to process (None = all)
            output_path: Path to save results JSON
            use_tracing: Enable MLflow tracing

        Returns:
            List of result dicts
        """
        data_path = data_path or self._settings.inference.data_path
        num_samples = num_samples or self._settings.inference.num_samples
        output_path = output_path or self._settings.inference.output_path

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        if self._model is None:
            self.setup()
        if self._schema_map is None:
            self.load_schemas()
        if use_tracing:
            self.setup_mlflow()

        data = []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                entry = json.loads(line)
                question = extract_question_from_input(entry.get("input", ""))
                db_id = entry.get("metadata", {}).get("db_id", "unknown")
                gold_sql = entry.get("output", "")
                data.append(
                    {
                        "question": question,
                        "db_id": db_id,
                        "gold_sql": gold_sql,
                    }
                )

        results = []
        generate_fn = self.generate_traced if use_tracing else self.generate

        for item in tqdm(data, desc="Generating SQL"):
            result = generate_fn(item["question"], item["db_id"])
            result["gold_sql"] = item["gold_sql"]
            result["exact_match"] = (
                result["sql"].lower().strip() == item["gold_sql"].lower().strip()
            )
            results.append(result)

        exact_matches = sum(1 for r in results if r["exact_match"])
        accuracy = exact_matches / len(results) if results else 0
        logger.info(
            f"Exact match accuracy: {accuracy:.2%} ({exact_matches}/{len(results)})"
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "metadata": {
                    "model": str(self._model_path),
                    "dataset": str(data_path),
                    "num_samples": len(results),
                    "accuracy": accuracy,
                    "timestamp": datetime.now().isoformat(),
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results
