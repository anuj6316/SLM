"""
Text2SQL Dataset Preprocessing Pipeline

Main entry point that orchestrates the full preprocessing workflow:
  1. Gather - Download datasets from HuggingFace
  2. Schema - Load or reverse-engineer database schemas
  3. Clean - Validate and normalize SQL queries
  4. Format - Convert to instruction-following format
  5. Output - Write to JSONL file
"""
import os
import json
import yaml
import logging
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

from src.gatherer import download_and_save
from src.cleaner import clean_and_validate
from src.formatter import normalize_to_sft
from src.schema_parser import load_spider_schemas
from src.schema_generator import generate_schema_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("main")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration file.

    Args:
        config_path: Path to config.yml

    Returns:
        Parsed configuration dict
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_type(filename: str) -> str:
    """
    Determines dataset type from filename.

    Args:
        filename: Name of the data file

    Returns:
        One of 'spider', 'bird', 'gretel', or 'unknown'
    """
    filename_lower = filename.lower()
    if "spider" in filename_lower:
        return "spider"
    if "bird" in filename_lower:
        return "bird"
    if "gretel" in filename_lower:
        return "gretel"
    return "unknown"


class ProcessingMetrics:
    """Tracks processing statistics per dataset."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"processed": 0, "dropped": 0, "errors": 0}
        )

    def record_success(self, dataset_type: str) -> None:
        self.stats[dataset_type]["processed"] += 1

    def record_dropped(self, dataset_type: str) -> None:
        self.stats[dataset_type]["dropped"] += 1

    def record_error(self, dataset_type: str) -> None:
        self.stats[dataset_type]["errors"] += 1

    def summary(self) -> str:
        lines = ["\n=== Processing Summary ==="]
        total_processed = 0
        total_dropped = 0
        total_errors = 0

        for ds_type, counts in sorted(self.stats.items()):
            lines.append(
                f"  {ds_type}: {counts['processed']} processed, "
                f"{counts['dropped']} dropped, {counts['errors']} errors"
            )
            total_processed += counts["processed"]
            total_dropped += counts["dropped"]
            total_errors += counts["errors"]

        lines.append(
            f"  TOTAL: {total_processed} processed, "
            f"{total_dropped} dropped, {total_errors} errors"
        )
        return "\n".join(lines)


def process_datasets(
    raw_files: List[str],
    output_file: str,
    spider_tables_path: str,
    buffer_size: int = 100
) -> int:
    """
    Processes raw dataset files into SFT-formatted JSONL.

    Args:
        raw_files: List of paths to raw JSONL files
        output_file: Path to output JSONL file
        spider_tables_path: Path to Spider tables.json
        buffer_size: Number of records to buffer before writing

    Returns:
        Total number of successfully processed records
    """
    schema_map = load_spider_schemas(spider_tables_path)

    if not schema_map:
        logger.info("External schemas not found or invalid. Triggering Reverse Engineering from raw data...")
        schema_map = generate_schema_map(raw_files)

    metrics = ProcessingMetrics()
    buffer: List[str] = []

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in raw_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            filename = os.path.basename(file_path)
            ds_type = get_dataset_type(filename)
            logger.info(f"--- Processing {filename} as {ds_type} ---")

            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, desc=f"Processing {ds_type}"):
                    try:
                        entry = json.loads(line)
                        cleaned = clean_and_validate(entry, ds_type)

                        if not cleaned:
                            metrics.record_dropped(ds_type)
                            continue

                        formatted = normalize_to_sft(cleaned, ds_type, schema_map)

                        if formatted:
                            buffer.append(json.dumps(formatted))
                            metrics.record_success(ds_type)

                            if len(buffer) >= buffer_size:
                                out_f.write('\n'.join(buffer) + '\n')
                                buffer.clear()
                        else:
                            metrics.record_dropped(ds_type)

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        metrics.record_error(ds_type)
                    except Exception as e:
                        logger.error(f"Error processing entry: {e}")
                        metrics.record_error(ds_type)

        if buffer:
            out_f.write('\n'.join(buffer) + '\n')

    logger.info(metrics.summary())
    total = sum(s["processed"] for s in metrics.stats.values())
    return total


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yml')
    cfg = load_config(config_path)

    logger.info(f"Configuration: {cfg}")

    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    raw_files = download_and_save(cfg['datasets'], output_dir=raw_dir)

    spider_tables = os.path.join(raw_dir, "spider_tables.json")
    output_sft = os.path.join(base_dir, "..", "data", "train_sft.jsonl")

    total_processed = process_datasets(raw_files, output_sft, spider_tables)
    logger.info(f"Pipeline complete. Output: {output_sft}")
