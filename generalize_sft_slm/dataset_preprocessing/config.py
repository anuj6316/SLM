"""
Configuration models for the tabular SFT pipeline.
Supports both programmatic construction and YAML-driven loading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import yaml


# ---------------------------------------------------------------------------
# Legacy dataclass (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class TabularDataset:
    """Minimal config for a single tabular data source."""
    type: str
    target_col: str
    ignore_cols: list
    path: str


# ---------------------------------------------------------------------------
# Production config models
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """Data source configuration."""
    type: Literal["csv", "excel", "database"]
    path: str                                   # file path or DB URI
    table_name: Optional[str] = None            # required for database source
    sheet_name: Optional[str] = None            # optional for Excel


@dataclass
class ColumnConfig:
    """Column selection and target configuration."""
    target: str                                 # target / label column
    ignore: list[str] = field(default_factory=list)
    feature_override: Optional[list[str]] = None  # explicit feature list; None = all non-target, non-ignored


@dataclass
class CleaningConfig:
    """Data cleaning options applied before formatting."""
    drop_nulls: bool = False                    # drop rows with any null
    fill_nulls: dict[str, Any] = field(default_factory=dict)  # {col: fill_value}
    drop_duplicates: bool = False
    strip_whitespace: bool = True               # strip leading/trailing whitespace from string cols


@dataclass
class FormattingConfig:
    """SFT output format configuration."""
    format: Literal["unsloth", "chatml", "alpaca"] = "unsloth"
    system_prompt: Optional[str] = None         # injected as system message (chatml/alpaca)
    instruction_template: Optional[str] = None  # custom instruction template; None = default


@dataclass
class SplitConfig:
    """Train / validation split configuration."""
    train_ratio: float = 0.9
    seed: int = 42
    stratify_by: Optional[str] = None           # column name for stratified split


@dataclass
class ExportConfig:
    """Output export configuration."""
    output_dir: str = "output"
    formats: list[Literal["jsonl", "huggingface", "parquet"]] = field(
        default_factory=lambda: ["jsonl"]
    )


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    source: SourceConfig
    columns: ColumnConfig
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ------------------------------------------------------------------
    # Factory: load from YAML file
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file.

        Expected YAML structure::

            source:
              type: csv
              path: /data/titanic.csv

            columns:
              target: Survived
              ignore: [PassengerId, Name]

            cleaning:
              drop_duplicates: true
              fill_nulls:
                Age: 0

            formatting:
              format: unsloth

            split:
              train_ratio: 0.9
              seed: 42

            export:
              output_dir: output
              formats: [jsonl, huggingface]
        """
        with open(path, "r") as fh:
            raw: dict = yaml.safe_load(fh)

        source_raw = raw.get("source", {})
        columns_raw = raw.get("columns", {})
        cleaning_raw = raw.get("cleaning", {})
        formatting_raw = raw.get("formatting", {})
        split_raw = raw.get("split", {})
        export_raw = raw.get("export", {})

        return cls(
            source=SourceConfig(**source_raw),
            columns=ColumnConfig(
                target=columns_raw["target"],
                ignore=columns_raw.get("ignore", []),
                feature_override=columns_raw.get("feature_override"),
            ),
            cleaning=CleaningConfig(
                drop_nulls=cleaning_raw.get("drop_nulls", False),
                fill_nulls=cleaning_raw.get("fill_nulls", {}),
                drop_duplicates=cleaning_raw.get("drop_duplicates", False),
                strip_whitespace=cleaning_raw.get("strip_whitespace", True),
            ),
            formatting=FormattingConfig(
                format=formatting_raw.get("format", "unsloth"),
                system_prompt=formatting_raw.get("system_prompt"),
                instruction_template=formatting_raw.get("instruction_template"),
            ),
            split=SplitConfig(
                train_ratio=split_raw.get("train_ratio", 0.9),
                seed=split_raw.get("seed", 42),
                stratify_by=split_raw.get("stratify_by"),
            ),
            export=ExportConfig(
                output_dir=export_raw.get("output_dir", "output"),
                formats=export_raw.get("formats", ["jsonl"]),
            ),
        )
