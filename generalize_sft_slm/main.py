"""
Entry point for the tabular SFT pipeline.

Usage (YAML-driven):
    python -m generalize_sft_slm.main --config generalize_sft_slm/config.yml

Usage (programmatic):
    python -m generalize_sft_slm.main
"""
import argparse
import logging
from pprint import pprint

from generalize_sft_slm.dataset_preprocessing import (
    PipelineConfig,
    SourceConfig,
    ColumnConfig,
    CleaningConfig,
    FormattingConfig,
    SplitConfig,
    ExportConfig,
)
from generalize_sft_slm.dataset_preprocessing.tabular_dataset import TabularSFTPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def run_from_yaml(config_path: str) -> None:
    """Run the pipeline from a YAML config file."""
    pipeline = TabularSFTPipeline.from_yaml(config_path)
    summary = pipeline.run()
    print("\n=== Pipeline Summary ===")
    pprint(summary)


def run_programmatic() -> None:
    """Run the pipeline with an inline programmatic config (example)."""
    cfg = PipelineConfig(
        source=SourceConfig(
            type="csv",
            path="data/Titanic-Dataset.csv",  # relative to project root
        ),
        columns=ColumnConfig(
            target="Survived",
            ignore=["PassengerId", "Name"],
        ),
        cleaning=CleaningConfig(
            drop_duplicates=True,
            fill_nulls={"Age": 0, "Cabin": "Unknown", "Embarked": "S"},
        ),
        formatting=FormattingConfig(format="unsloth"),
        split=SplitConfig(train_ratio=0.9, seed=42, stratify_by="Survived"),
        export=ExportConfig(output_dir="output", formats=["jsonl", "huggingface"]),
    )
    pipeline = TabularSFTPipeline(cfg)
    summary = pipeline.run()
    print("\n=== Pipeline Summary ===")
    pprint(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular SFT Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If omitted, runs the built-in example.",
    )
    args = parser.parse_args()

    if args.config:
        run_from_yaml(args.config)
    else:
        run_programmatic()
