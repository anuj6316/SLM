"""
TrainValSplitter: split a HuggingFace Dataset into train and validation subsets.

Supports:
  - Random split
  - Stratified split (by a column present in the dataset)
"""
from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, DatasetDict

from ..config import SplitConfig

logger = logging.getLogger(__name__)


class TrainValSplitter:
    """Split a HuggingFace Dataset into train / validation subsets."""

    def __init__(self, split_cfg: SplitConfig):
        self.split_cfg = split_cfg

    def split(self, dataset: Dataset) -> DatasetDict:
        """Return a DatasetDict with 'train' and 'validation' keys."""
        cfg = self.split_cfg
        val_ratio = round(1.0 - cfg.train_ratio, 10)

        logger.info(
            f"Splitting {len(dataset)} rows — "
            f"train: {cfg.train_ratio:.0%}, val: {val_ratio:.0%}, "
            f"seed: {cfg.seed}, stratify_by: {cfg.stratify_by!r}"
        )

        split_kwargs: dict = {
            "test_size": val_ratio,
            "seed": cfg.seed,
            "shuffle": True,
        }

        if cfg.stratify_by and cfg.stratify_by in dataset.column_names:
            split_kwargs["stratify_by_column"] = cfg.stratify_by
            logger.info(f"Using stratified split on column '{cfg.stratify_by}'")
        elif cfg.stratify_by:
            logger.warning(
                f"stratify_by column '{cfg.stratify_by}' not found in dataset "
                f"(columns: {dataset.column_names}). Falling back to random split."
            )

        splits = dataset.train_test_split(**split_kwargs)

        result = DatasetDict(
            train=splits["train"],
            validation=splits["test"],
        )
        logger.info(
            f"Split complete — train: {len(result['train'])}, "
            f"validation: {len(result['validation'])}"
        )
        return result
