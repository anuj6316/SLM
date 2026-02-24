"""
Gradio UI for the Tabular SFT Pipeline.

Launch:
    python generalize_sft_slm/app.py
    # or
    python -m generalize_sft_slm.app

The UI exposes every PipelineConfig knob across five accordion sections:
  1. Data Source      — file upload (CSV/Excel) or DB URI + table name
  2. Columns          — target column, ignore columns, optional feature override
  3. Cleaning         — null handling, dedup, whitespace stripping
  4. Formatting       — SFT format, system prompt, instruction template
  5. Split & Export   — train ratio, seed, stratify column, output dir, formats

After running the pipeline the UI shows:
  • A summary table (rows loaded / cleaned / train / val, elapsed time)
  • A live preview of the first 5 formatted training examples
  • Download buttons for every written file
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import polars as pl

# ---------------------------------------------------------------------------
# Make the package importable when the script is run directly from the repo
# root or from inside the generalize_sft_slm/ directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # generalize_sft_slm/
_ROOT = _HERE.parent                             # project root
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generalize_sft_slm.dataset_preprocessing import (  # noqa: E402
    CleaningConfig,
    ColumnConfig,
    ExportConfig,
    FormattingConfig,
    PipelineConfig,
    SourceConfig,
    SplitConfig,
)
from generalize_sft_slm.dataset_preprocessing.tabular_dataset import (  # noqa: E402
    TabularSFTPipeline,
)

# ---------------------------------------------------------------------------
# Logging — capture pipeline logs and surface them in the UI
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: peek at uploaded file columns
# ---------------------------------------------------------------------------

def _get_columns_from_file(file_obj) -> list[str]:
    """Return column names from an uploaded CSV or Excel file."""
    if file_obj is None:
        return []
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext == ".csv":
            df = pl.read_csv(path, n_rows=1)
            return df.columns
        elif ext in [".xls", ".xlsx"]:
            import pandas as pd
            df = pd.read_excel(path, nrows=1)
            return list(df.columns)
    except Exception:
        pass
    return []


def refresh_columns(file_obj):
    """Called when a file is uploaded — populate column dropdowns."""
    cols = _get_columns_from_file(file_obj)
    if not cols:
        return (
            gr.update(choices=[], value=None),   # target_col
            gr.update(choices=[], value=[]),      # ignore_cols
            gr.update(choices=[], value=[]),      # feature_override
            gr.update(choices=[], value=None),    # stratify_by
        )
    return (
        gr.update(choices=cols, value=cols[0] if cols else None),
        gr.update(choices=cols, value=[]),
        gr.update(choices=cols, value=[]),
        gr.update(choices=["(none)"] + cols, value="(none)"),
    )


# ---------------------------------------------------------------------------
# Core: run the pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    # Source
    source_type: str,
    uploaded_file,
    db_uri: str,
    db_table: str,
    # Columns
    target_col: str,
    ignore_cols: list[str],
    feature_override: list[str],
    # Cleaning
    drop_nulls: bool,
    fill_nulls_json: str,
    drop_duplicates: bool,
    strip_whitespace: bool,
    # Formatting
    sft_format: str,
    system_prompt: str,
    instruction_template: str,
    # Split
    train_ratio: float,
    seed: int,
    stratify_by: str,
    # Export
    output_dir: str,
    export_jsonl: bool,
    export_hf: bool,
    export_parquet: bool,
):
    """Build PipelineConfig from UI inputs and run the pipeline."""

    log_lines: list[str] = []

    # ---- Validate inputs --------------------------------------------------
    if source_type in ("csv", "excel"):
        if uploaded_file is None:
            return "❌ Please upload a CSV or Excel file.", "", [], None
        file_path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    else:  # database
        if not db_uri.strip():
            return "❌ Please provide a database URI.", "", [], None
        if not db_table.strip():
            return "❌ Please provide a table name.", "", [], None
        file_path = db_uri.strip()

    if not target_col:
        return "❌ Please select a target column.", "", [], None

    # ---- Parse fill_nulls JSON -------------------------------------------
    fill_nulls: dict = {}
    if fill_nulls_json.strip():
        try:
            fill_nulls = json.loads(fill_nulls_json)
            if not isinstance(fill_nulls, dict):
                return "❌ fill_nulls must be a JSON object, e.g. {\"Age\": 0}", "", [], None
        except json.JSONDecodeError as exc:
            return f"❌ Invalid fill_nulls JSON: {exc}", "", [], None

    # ---- Export formats --------------------------------------------------
    formats: list[str] = []
    if export_jsonl:
        formats.append("jsonl")
    if export_hf:
        formats.append("huggingface")
    if export_parquet:
        formats.append("parquet")
    if not formats:
        formats = ["jsonl"]

    # ---- Resolve output dir (relative to project root) -------------------
    out_dir = output_dir.strip() or "generalize_sft_slm/output"

    # ---- Build config ----------------------------------------------------
    try:
        cfg = PipelineConfig(
            source=SourceConfig(
                type=source_type,
                path=file_path,
                table_name=db_table.strip() or None,
            ),
            columns=ColumnConfig(
                target=target_col,
                ignore=list(ignore_cols) if ignore_cols else [],
                feature_override=list(feature_override) if feature_override else None,
            ),
            cleaning=CleaningConfig(
                drop_nulls=drop_nulls,
                fill_nulls=fill_nulls,
                drop_duplicates=drop_duplicates,
                strip_whitespace=strip_whitespace,
            ),
            formatting=FormattingConfig(
                format=sft_format,
                system_prompt=system_prompt.strip() or None,
                instruction_template=instruction_template.strip() or None,
            ),
            split=SplitConfig(
                train_ratio=float(train_ratio),
                seed=int(seed),
                stratify_by=stratify_by if stratify_by and stratify_by != "(none)" else None,
            ),
            export=ExportConfig(
                output_dir=out_dir,
                formats=formats,
            ),
        )
    except Exception as exc:
        return f"❌ Config error: {exc}", "", [], None

    # ---- Run pipeline ----------------------------------------------------
    # Capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        pipeline = TabularSFTPipeline(cfg)
        summary = pipeline.run()
    except Exception as exc:
        root_logger.removeHandler(handler)
        tb = traceback.format_exc()
        return f"❌ Pipeline failed:\n\n{tb}", log_stream.getvalue(), [], None
    finally:
        root_logger.removeHandler(handler)

    logs = log_stream.getvalue()

    # ---- Build summary markdown ------------------------------------------
    written = summary.get("written_files", {})
    written_lines = []
    for fmt, paths in written.items():
        for p in paths:
            written_lines.append(f"  • `{p}`")
    written_str = "\n".join(written_lines) if written_lines else "  *(none)*"

    summary_md = f"""## ✅ Pipeline Complete

| Metric | Value |
|--------|-------|
| Rows loaded | **{summary['rows_loaded']:,}** |
| Rows after cleaning | **{summary['rows_after_clean']:,}** |
| Train rows | **{summary['train_rows']:,}** |
| Validation rows | **{summary['val_rows']:,}** |
| Elapsed | **{summary['elapsed_sec']}s** |

### Written files
{written_str}
"""

    # ---- Preview: first 5 training examples ------------------------------
    preview_md = _build_preview(out_dir, sft_format, formats)

    # ---- Download file list ----------------------------------------------
    download_paths = []
    for fmt, paths in written.items():
        for p in paths:
            if os.path.isfile(p):
                download_paths.append(p)

    return summary_md, logs, download_paths, preview_md


def _build_preview(out_dir: str, sft_format: str, formats: list[str]) -> str:
    """Read the first 5 rows from the written train split and render as markdown."""
    # Try JSONL first (most readable), then HuggingFace Dataset
    jsonl_path = os.path.join(out_dir, "train.jsonl")
    if "jsonl" in formats and os.path.isfile(jsonl_path):
        try:
            rows = []
            with open(jsonl_path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 5:
                        break
                    rows.append(json.loads(line))
            if not rows:
                return "*No rows to preview.*"
            lines = [f"### Preview — first {len(rows)} training examples (JSONL)\n"]
            for i, row in enumerate(rows, 1):
                lines.append(f"**Example {i}**\n```json\n{json.dumps(row, indent=2, ensure_ascii=False)}\n```\n")
            return "\n".join(lines)
        except Exception as exc:
            return f"*Preview unavailable: {exc}*"

    # Fallback: HuggingFace Dataset
    hf_path = os.path.join(out_dir, "hf_dataset")
    if "huggingface" in formats and os.path.isdir(hf_path):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(hf_path)
            train_ds = ds["train"] if hasattr(ds, "__getitem__") else ds
            rows = [train_ds[i] for i in range(min(5, len(train_ds)))]
            lines = [f"### Preview — first {len(rows)} training examples (HuggingFace Dataset)\n"]
            for i, row in enumerate(rows, 1):
                lines.append(f"**Example {i}**\n```json\n{json.dumps(row, indent=2, ensure_ascii=False)}\n```\n")
            return "\n".join(lines)
        except Exception as exc:
            return f"*Preview unavailable: {exc}*"

    return "*Preview not available for the selected export format.*"


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_CSS = """
.section-header { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
.run-btn { background: #2563eb !important; color: white !important; font-size: 1rem !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Tabular SFT Pipeline", css=_CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🗂️ Tabular → SFT Dataset Pipeline\n"
        "Configure every step of the pipeline, upload your data, and export "
        "train/validation splits in **Unsloth**, **ChatML**, or **Alpaca** format."
    )

    with gr.Row():
        # ------------------------------------------------------------------ LEFT COLUMN
        with gr.Column(scale=1):

            # ---- 1. Data Source ------------------------------------------
            with gr.Accordion("📂 1 · Data Source", open=True):
                source_type = gr.Radio(
                    choices=["csv", "excel", "database"],
                    value="csv",
                    label="Source type",
                )
                with gr.Group() as file_group:
                    uploaded_file = gr.File(
                        label="Upload CSV / Excel file",
                        file_types=[".csv", ".xls", ".xlsx"],
                    )
                with gr.Group(visible=False) as db_group:
                    db_uri = gr.Textbox(
                        label="Database URI",
                        placeholder="sqlite:///chinook.db  or  mysql+pymysql://user:pass@host/db",
                    )
                    db_table = gr.Textbox(label="Table name", placeholder="albums")

                def _toggle_source(src):
                    is_file = src in ("csv", "excel")
                    return gr.update(visible=is_file), gr.update(visible=not is_file)

                source_type.change(_toggle_source, source_type, [file_group, db_group])

            # ---- 2. Columns ----------------------------------------------
            with gr.Accordion("🎯 2 · Columns", open=True):
                target_col = gr.Dropdown(
                    choices=[], label="Target column", allow_custom_value=True
                )
                ignore_cols = gr.Dropdown(
                    choices=[], label="Ignore columns (multi-select)",
                    multiselect=True, allow_custom_value=True,
                )
                feature_override = gr.Dropdown(
                    choices=[],
                    label="Feature override (leave empty = all non-target, non-ignored)",
                    multiselect=True, allow_custom_value=True,
                )
                uploaded_file.change(
                    refresh_columns,
                    inputs=uploaded_file,
                    outputs=[target_col, ignore_cols, feature_override, gr.State()],
                )

            # ---- 3. Cleaning ---------------------------------------------
            with gr.Accordion("🧹 3 · Cleaning", open=False):
                drop_nulls = gr.Checkbox(label="Drop rows with any null", value=False)
                fill_nulls_json = gr.Textbox(
                    label='Fill nulls (JSON object, e.g. {"Age": 0, "Cabin": "Unknown"})',
                    placeholder='{"Age": 0, "Cabin": "Unknown", "Embarked": "S"}',
                    lines=2,
                )
                drop_duplicates = gr.Checkbox(label="Drop duplicate rows", value=True)
                strip_whitespace = gr.Checkbox(label="Strip whitespace from string columns", value=True)

            # ---- 4. Formatting -------------------------------------------
            with gr.Accordion("✍️ 4 · SFT Formatting", open=False):
                sft_format = gr.Radio(
                    choices=["unsloth", "chatml", "alpaca"],
                    value="unsloth",
                    label="Output format",
                )
                system_prompt = gr.Textbox(
                    label="System prompt (leave empty for built-in default)",
                    placeholder="You are a data analysis assistant...",
                    lines=3,
                )
                instruction_template = gr.Textbox(
                    label="Instruction template (use {target} and {features} placeholders; leave empty for default)",
                    placeholder="Analyze the following data features and predict the exact value for {target}.\n\nData Features:\n{features}",
                    lines=4,
                )

            # ---- 5. Split & Export ---------------------------------------
            with gr.Accordion("✂️ 5 · Split & Export", open=False):
                train_ratio = gr.Slider(
                    minimum=0.5, maximum=0.99, step=0.01, value=0.9,
                    label="Train ratio",
                )
                seed = gr.Number(value=42, label="Random seed", precision=0)
                stratify_by = gr.Dropdown(
                    choices=["(none)"], value="(none)",
                    label="Stratify split by column",
                    allow_custom_value=True,
                )
                # Wire file upload → stratify dropdown too
                uploaded_file.change(
                    refresh_columns,
                    inputs=uploaded_file,
                    outputs=[
                        gr.State(),   # target_col (already wired above; duplicate outputs are fine)
                        gr.State(),   # ignore_cols
                        gr.State(),   # feature_override
                        stratify_by,
                    ],
                )
                output_dir = gr.Textbox(
                    label="Output directory",
                    value="generalize_sft_slm/output",
                )
                with gr.Row():
                    export_jsonl = gr.Checkbox(label="JSONL", value=True)
                    export_hf = gr.Checkbox(label="HuggingFace Dataset", value=True)
                    export_parquet = gr.Checkbox(label="Parquet", value=False)

            run_btn = gr.Button("▶ Run Pipeline", variant="primary", elem_classes="run-btn")

        # ------------------------------------------------------------------ RIGHT COLUMN
        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Summary", value="*Run the pipeline to see results.*")

            with gr.Accordion("📋 Preview (first 5 training examples)", open=True):
                preview_out = gr.Markdown(value="*Run the pipeline to see a preview.*")

            with gr.Accordion("📜 Pipeline logs", open=False):
                logs_out = gr.Textbox(
                    label="Logs",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )

            with gr.Accordion("⬇️ Download outputs", open=True):
                download_out = gr.Files(label="Written files")

    # ---- Wire run button -------------------------------------------------
    run_btn.click(
        fn=run_pipeline,
        inputs=[
            source_type, uploaded_file, db_uri, db_table,
            target_col, ignore_cols, feature_override,
            drop_nulls, fill_nulls_json, drop_duplicates, strip_whitespace,
            sft_format, system_prompt, instruction_template,
            train_ratio, seed, stratify_by,
            output_dir, export_jsonl, export_hf, export_parquet,
        ],
        outputs=[summary_out, logs_out, download_out, preview_out],
    )

    gr.Markdown(
        "---\n"
        "*Powered by [Polars](https://pola.rs) · "
        "[HuggingFace Datasets](https://huggingface.co/docs/datasets) · "
        "[Gradio](https://gradio.app)*"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
