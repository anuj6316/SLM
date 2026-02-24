"""
SFT formatters: convert a cleaned Polars DataFrame into a HuggingFace Dataset
in one of three conversation formats — Unsloth, ChatML, or Alpaca.

All formatters use vectorised Polars operations for performance.
"""
from __future__ import annotations

import logging
from typing import Optional

import polars as pl
from datasets import Dataset

from ..config import ColumnConfig, FormattingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUCTION = (
    "Analyze the following data features and predict the exact value for {target}.\n\n"
    "Data Features:\n{features}"
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a data analysis assistant. "
    "Given a set of tabular features, predict the target value accurately."
)


# ---------------------------------------------------------------------------
# Internal helper: build feature text + human/ai columns via Polars
# ---------------------------------------------------------------------------

def _build_feature_columns(
    df: pl.DataFrame,
    column_cfg: ColumnConfig,
    instruction_template: Optional[str] = None,
) -> pl.DataFrame:
    """Return df with two new columns: 'human_input' and 'ai_output'."""

    target = column_cfg.target
    ignore = set(column_cfg.ignore)

    if column_cfg.feature_override:
        feature_cols = [c for c in column_cfg.feature_override if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c != target and c not in ignore]

    # Build "- col: value" lines for every feature column
    feature_exprs = [
        pl.format("- {}: {}", pl.lit(col), pl.col(col))
        for col in feature_cols
    ]
    df = df.with_columns(
        pl.concat_str(feature_exprs, separator="\n").alias("_features_text")
    )

    # Build human instruction using Python str.format_map so that {target} and
    # {features} can appear in any order in a custom template without silently
    # swapping arguments.
    template = instruction_template or _DEFAULT_INSTRUCTION
    df = df.with_columns(
        pl.struct(["_features_text"]).map_elements(
            lambda row: template.format(target=target, features=row["_features_text"]),
            return_dtype=pl.String,
        ).alias("_human_input"),
        pl.col(target).cast(pl.String).alias("_ai_output"),
    )

    return df.drop("_features_text")


# ---------------------------------------------------------------------------
# Public formatters
# ---------------------------------------------------------------------------

def _format_for_unsloth(
    clf,                        # TabularDataset (legacy) or ColumnConfig
    df: pl.DataFrame,
    columns: list[str],         # kept for backward-compat; unused internally
    formatting_cfg: Optional[FormattingConfig] = None,
) -> Dataset:
    """Unsloth conversations format.

    Output schema::

        {
          "conversations": [
            {"from": "human",     "content": "<instruction>"},
            {"from": "assistant", "content": "<label>"}
          ]
        }
    """
    # Support both legacy TabularDataset and new ColumnConfig
    if hasattr(clf, "target_col"):
        # Legacy TabularDataset
        from ..config import ColumnConfig
        column_cfg = ColumnConfig(target=clf.target_col, ignore=clf.ignore_cols or [])
    else:
        column_cfg = clf  # already a ColumnConfig

    instruction_template = (
        formatting_cfg.instruction_template if formatting_cfg else None
    )
    df = _build_feature_columns(df, column_cfg, instruction_template)

    conversations = [
        [
            {"from": "human", "content": h},
            {"from": "assistant", "content": a},
        ]
        for h, a in zip(df["_human_input"], df["_ai_output"])
    ]
    logger.info(f"Formatted {len(conversations)} rows → Unsloth conversations")
    return Dataset.from_dict({"conversations": conversations})


def _format_for_chatml(
    column_cfg: ColumnConfig,
    df: pl.DataFrame,
    formatting_cfg: Optional[FormattingConfig] = None,
) -> Dataset:
    """ChatML messages format.

    Output schema::

        {
          "messages": [
            {"role": "system",    "content": "<system_prompt>"},
            {"role": "user",      "content": "<instruction>"},
            {"role": "assistant", "content": "<label>"}
          ]
        }
    """
    system_prompt = (
        (formatting_cfg.system_prompt if formatting_cfg else None)
        or _DEFAULT_SYSTEM_PROMPT
    )
    instruction_template = (
        formatting_cfg.instruction_template if formatting_cfg else None
    )
    df = _build_feature_columns(df, column_cfg, instruction_template)

    messages = [
        [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": h},
            {"role": "assistant", "content": a},
        ]
        for h, a in zip(df["_human_input"], df["_ai_output"])
    ]
    logger.info(f"Formatted {len(messages)} rows → ChatML messages")
    return Dataset.from_dict({"messages": messages})


def _format_for_alpaca(
    column_cfg: ColumnConfig,
    df: pl.DataFrame,
    formatting_cfg: Optional[FormattingConfig] = None,
) -> Dataset:
    """Alpaca instruction / input / output format.

    Output schema::

        {
          "instruction": "<system_prompt>",
          "input":       "<feature description>",
          "output":      "<label>"
        }
    """
    system_prompt = (
        (formatting_cfg.system_prompt if formatting_cfg else None)
        or _DEFAULT_SYSTEM_PROMPT
    )
    instruction_template = (
        formatting_cfg.instruction_template if formatting_cfg else None
    )
    df = _build_feature_columns(df, column_cfg, instruction_template)

    records = {
        "instruction": [system_prompt] * len(df),
        "input":       list(df["_human_input"]),
        "output":      list(df["_ai_output"]),
    }
    logger.info(f"Formatted {len(df)} rows → Alpaca instruction/input/output")
    return Dataset.from_dict(records)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def format_dataset(
    column_cfg: ColumnConfig,
    df: pl.DataFrame,
    formatting_cfg: FormattingConfig,
) -> Dataset:
    """Route to the correct formatter based on FormattingConfig.format."""
    fmt = formatting_cfg.format
    if fmt == "unsloth":
        return _format_for_unsloth(column_cfg, df, df.columns, formatting_cfg)
    elif fmt == "chatml":
        return _format_for_chatml(column_cfg, df, formatting_cfg)
    elif fmt == "alpaca":
        return _format_for_alpaca(column_cfg, df, formatting_cfg)
    else:
        raise ValueError(
            f"Unknown format '{fmt}'. Supported: 'unsloth', 'chatml', 'alpaca'."
        )
