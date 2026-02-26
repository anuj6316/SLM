"""
TabularSFTPipeline: end-to-end orchestrator for the tabular → SFT pipeline.

Flow:
  1. Load   — ingest CSV / Excel / Database into a Polars DataFrame
  2. Clean  — drop ignored cols, handle nulls, dedup, strip whitespace
  3. Format — convert to Unsloth / ChatML / Alpaca HuggingFace Dataset
  4. Split  — train / validation split
  5. Export — write JSONL / HuggingFace Dataset / Parquet to disk

Usage (programmatic)::

    from generalize_sft_slm.dataset_preprocessing import (
        PipelineConfig, SourceConfig, ColumnConfig, TabularSFTPipeline
    )

    cfg = PipelineConfig(
        source=SourceConfig(type="csv", path="data/titanic.csv"),
        columns=ColumnConfig(target="Survived", ignore=["PassengerId", "Name"]),
    )
    result = TabularSFTPipeline(cfg).run()

Usage (YAML-driven)::

    result = TabularSFTPipeline.from_yaml("generalize_sft_slm/config.yml").run()
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import polars as pl
from datasets import DatasetDict

from ..config import PipelineConfig
from .cleaner import TabularCleaner
from .data_ingestion import ProcessDatabase, ProcessLocalFile
from .exporter import Exporter
from .splitter import TrainValSplitter
from .utils import format_dataset

logger = logging.getLogger(__name__)


class TabularSFTPipeline:
    """Orchestrate the full tabular → SFT dataset pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Factory: YAML-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TabularSFTPipeline":
        """Construct a pipeline from a YAML config file."""
        cfg = PipelineConfig.from_yaml(yaml_path)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pipeline and return a summary dict.

        Returns::

            {
                "rows_loaded":    int,
                "rows_after_clean": int,
                "train_rows":     int,
                "val_rows":       int,
                "written_files":  dict[format, list[path]],
                "elapsed_sec":    float,
            }
        """
        t0 = time.perf_counter()
        cfg = self.config

        # 1. Load -------------------------------------------------------
        logger.info("=== Step 1/5: Loading data ===")
        df = self._load()
        rows_loaded = len(df)
        logger.info(f"Loaded {rows_loaded} rows, {len(df.columns)} columns")

        # 2. Clean ------------------------------------------------------
        logger.info("=== Step 2/5: Cleaning data ===")
        cleaner = TabularCleaner(cfg.columns, cfg.cleaning)
        df = cleaner.clean(df)
        rows_clean = len(df)

        # 3. Format -----------------------------------------------------
        logger.info(f"=== Step 3/5: Formatting as '{cfg.formatting.format}' ===")
        dataset = format_dataset(cfg.columns, df, cfg.formatting)

        # 4. Split ------------------------------------------------------
        logger.info("=== Step 4/5: Splitting train / validation ===")
        splitter = TrainValSplitter(cfg.split)
        dataset_dict: DatasetDict = splitter.split(dataset)

        # 5. Export -----------------------------------------------------
        logger.info("=== Step 5/5: Exporting ===")
        exporter = Exporter(cfg.export)
        written = exporter.export(dataset_dict)

        elapsed = round(time.perf_counter() - t0, 2)
        summary = {
            "rows_loaded":      rows_loaded,
            "rows_after_clean": rows_clean,
            "train_rows":       len(dataset_dict["train"]),
            "val_rows":         len(dataset_dict["validation"]),
            "written_files":    written,
            "elapsed_sec":      elapsed,
        }
        logger.info(f"Pipeline complete in {elapsed}s — summary: {summary}")
        return summary

    # ------------------------------------------------------------------
    # Private: load step
    # ------------------------------------------------------------------

    def _load(self) -> pl.DataFrame:
        src = self.config.source

        if src.type == "database":
            if not src.table_name:
                raise ValueError("SourceConfig.table_name is required for database sources.")
            processor = ProcessDatabase(
                connectivity_uri=src.path,
                table_name=src.table_name,
                target_col=self.config.columns.target,
                ignore_col=self.config.columns.ignore,
            )
            pd_df: pd.DataFrame = processor.fetch_filtered_data()
            return pl.from_pandas(pd_df)

        else:
            # CSV or Excel — use ProcessLocalFile via a lightweight adapter
            from ..config import TabularDataset
            legacy_cfg = TabularDataset(
                type=src.type,
                target_col=self.config.columns.target,
                ignore_cols=self.config.columns.ignore,
                path=src.path,
            )
            loader = ProcessLocalFile(legacy_cfg)
            return loader.df  # already a Polars DataFrame
