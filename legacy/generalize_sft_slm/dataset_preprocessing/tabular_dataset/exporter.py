"""
Exporter: write a DatasetDict (train / validation) to one or more output formats.

Supported formats:
  - jsonl        → <output_dir>/train.jsonl, <output_dir>/val.jsonl
  - huggingface  → <output_dir>/hf_dataset/  (HuggingFace Dataset on disk)
  - parquet      → <output_dir>/train.parquet, <output_dir>/val.parquet
"""
from __future__ import annotations

import logging
import os

from datasets import DatasetDict

from ..config import ExportConfig

logger = logging.getLogger(__name__)


class Exporter:
    """Write a DatasetDict to one or more output formats."""

    def __init__(self, export_cfg: ExportConfig):
        self.export_cfg = export_cfg

    def export(self, dataset_dict: DatasetDict) -> dict[str, list[str]]:
        """Export all splits to all configured formats.

        Returns a dict mapping format name → list of written file paths.
        """
        os.makedirs(self.export_cfg.output_dir, exist_ok=True)
        written: dict[str, list[str]] = {}

        for fmt in self.export_cfg.formats:
            if fmt == "jsonl":
                paths = self._export_jsonl(dataset_dict)
            elif fmt == "huggingface":
                paths = self._export_huggingface(dataset_dict)
            elif fmt == "parquet":
                paths = self._export_parquet(dataset_dict)
            else:
                logger.warning(f"Unknown export format '{fmt}' — skipping.")
                continue
            written[fmt] = paths

        return written

    # ------------------------------------------------------------------
    # Format implementations
    # ------------------------------------------------------------------

    def _export_jsonl(self, dataset_dict: DatasetDict) -> list[str]:
        paths = []
        split_map = {"train": "train.jsonl", "validation": "val.jsonl"}
        for split, filename in split_map.items():
            if split not in dataset_dict:
                continue
            path = os.path.join(self.export_cfg.output_dir, filename)
            dataset_dict[split].to_json(path, force_ascii=False)
            logger.info(f"Wrote JSONL → {path} ({len(dataset_dict[split])} rows)")
            paths.append(path)
        return paths

    def _export_huggingface(self, dataset_dict: DatasetDict) -> list[str]:
        path = os.path.join(self.export_cfg.output_dir, "hf_dataset")
        dataset_dict.save_to_disk(path)
        logger.info(f"Saved HuggingFace Dataset → {path}")
        return [path]

    def _export_parquet(self, dataset_dict: DatasetDict) -> list[str]:
        paths = []
        split_map = {"train": "train.parquet", "validation": "val.parquet"}
        for split, filename in split_map.items():
            if split not in dataset_dict:
                continue
            path = os.path.join(self.export_cfg.output_dir, filename)
            dataset_dict[split].to_parquet(path)
            logger.info(f"Wrote Parquet → {path} ({len(dataset_dict[split])} rows)")
            paths.append(path)
        return paths
