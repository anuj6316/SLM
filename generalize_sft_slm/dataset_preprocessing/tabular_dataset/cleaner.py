"""
TabularCleaner: applies a configurable chain of cleaning operations to a Polars DataFrame.

Operations (in order):
  1. Drop ignored columns
  2. Fill / drop nulls
  3. Remove duplicate rows
  4. Strip whitespace from string columns
  5. Validate that the target column exists and has no nulls
"""
from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from ..config import CleaningConfig, ColumnConfig

logger = logging.getLogger(__name__)


class TabularCleaner:
    """Apply a configurable cleaning chain to a Polars DataFrame."""

    def __init__(self, column_cfg: ColumnConfig, cleaning_cfg: CleaningConfig):
        self.column_cfg = column_cfg
        self.cleaning_cfg = cleaning_cfg

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run the full cleaning chain and return the cleaned DataFrame."""
        logger.info(f"Cleaning started — shape: {df.shape}")

        df = self._drop_ignored_columns(df)
        df = self._handle_nulls(df)
        df = self._drop_duplicates(df)
        df = self._strip_whitespace(df)
        self._validate_target(df)

        logger.info(f"Cleaning complete — shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _drop_ignored_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_drop = [c for c in self.column_cfg.ignore if c in df.columns]
        if cols_to_drop:
            logger.info(f"Dropping ignored columns: {cols_to_drop}")
            df = df.drop(cols_to_drop)
        return df

    def _handle_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.cleaning_cfg

        # Fill specific columns first
        if cfg.fill_nulls:
            fill_exprs = []
            for col, fill_val in cfg.fill_nulls.items():
                if col in df.columns:
                    fill_exprs.append(pl.col(col).fill_null(fill_val))
            if fill_exprs:
                df = df.with_columns(fill_exprs)
                logger.info(f"Filled nulls in columns: {list(cfg.fill_nulls.keys())}")

        # Then drop rows that still have nulls
        if cfg.drop_nulls:
            before = len(df)
            df = df.drop_nulls()
            dropped = before - len(df)
            if dropped:
                logger.info(f"Dropped {dropped} rows containing nulls")

        return df

    def _drop_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cleaning_cfg.drop_duplicates:
            before = len(df)
            df = df.unique()
            dropped = before - len(df)
            if dropped:
                logger.info(f"Dropped {dropped} duplicate rows")
        return df

    def _strip_whitespace(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.cleaning_cfg.strip_whitespace:
            return df
        string_cols = [c for c in df.columns if df[c].dtype == pl.String]
        if string_cols:
            df = df.with_columns([pl.col(c).str.strip_chars() for c in string_cols])
        return df

    def _validate_target(self, df: pl.DataFrame) -> None:
        target = self.column_cfg.target
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )
        null_count = df[target].null_count()
        if null_count > 0:
            logger.warning(
                f"Target column '{target}' has {null_count} null values. "
                "These rows will produce empty labels."
            )
