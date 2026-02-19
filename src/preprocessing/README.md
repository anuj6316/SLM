# Dataset Preprocessing Module

Modular, scalable pipeline for converting datasets into instruction-following JSONL format for fine-tuning language models.

## 📁 Module Structure

```
src __init__.py/preprocessing/
├──
├── base.py                      # Abstract base classes
├── dataset_preprocessor.py      # Main orchestrator
├── loaders/                     # Data source plugins
│   ├── __init__.py
│   ├── csv_loader.py
│   ├── excel_loader.py
│   └── hf_loader.py
├── processors/                  # Transformation plugins
│   ├── __init__.py
│   ├── column_filter.py
│   ├── instruction_builder.py
│   └── schema_injector.py
└── formatters/                 # Output format plugins
    ├── __init__.py
    ├── alpaca_formatter.py
    └── chatml_formatter.py
```

## 🔄 Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET PREPROCESSING FLOW                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD DATA          2. PROCESS          3. FORMAT            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ CSV/Excel/HF │───▶│ Filter Cols │───▶│ Instruction/│       │
│  │   Dataset     │    │ Add Context │    │   Output     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                    │              │
│                                                    ▼              │
│                                              ┌──────────────┐     │
│                                              │   JSONL      │     │
│                                              │   Output     │     │
│                                              └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## 📖 Usage

### 1. Configure in `config.yaml`

```yaml
preprocessing:
  # Data Source
  source:
    type: "hf"  # csv, excel, or hf
    path: "YOUR_HF_DATASET_NAME"  # or local file path

  # Column Configuration
  columns:
    ignore: ["id", "timestamp", "metadata_column"]  # Columns to skip
    input: ["question", "context"]                   # Input fields
    output: ["sql_query"]                           # Target field
    system_prompt: "Convert this table data to SQL" # Optional override

  # Transformations
  transformations:
    add_context: true
    context_columns: ["table_schema", "description"]

  # Output Format
  output:
    format: "instruction_output"  # or "chatml"
    path: "data/processed/train_sft.jsonl"
```

### 2. Run Preprocessing

```bash
python main.py prepare
```

## 🔧 Configuration Options

### Source Options

| Option | Type | Description |
|--------|------|-------------|
| `type` | string | Data source type: `csv`, `excel`, or `hf` |
| `path` | string | File path or HuggingFace dataset name |
| `split` | string | HF dataset split (train, validation, test) |

### Column Options

| Option | Type | Description |
|--------|------|-------------|
| `ignore` | list | Column names to exclude |
| `input` | list | Input column names (concatenated) |
| `output` | list | Output column names (target) |
| `system_prompt` | string | Override system prompt |

### Output Formats

#### `instruction_output` (Default)
```json
{"instruction": "...", "output": "..."}
```

#### `chatml`
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 🏗️ Architecture

### Base Classes

#### `BaseLoader`
Abstract interface for all data loaders.

```python
class BaseLoader:
    def load(self, config: dict) -> Generator[dict, None, None]:
        """Load dataset as stream of examples."""
```

#### `BaseProcessor`
Interface for transformation processors.

```python
class BaseProcessor:
    def process(self, example: dict, config: dict) -> dict:
        """Transform a single example."""
```

#### `BaseFormatter`
Interface for output formatting.

```python
class BaseFormatter:
    def format(self, example: dict, config: dict) -> dict:
        """Format example to target JSONL structure."""
```

### Plugin System

Each subdirectory (`loaders/`, `processors/`, `formatters/`) implements plugins:

- **Loaders**: `CsvLoader`, `ExcelLoader`, `HfLoader`
- **Processors**: `ColumnFilter`, `InstructionBuilder`, `SchemaInjector`
- **Formatters**: `AlpacaFormatter`, `ChatMLFormatter`

## 📦 Adding New Plugins

### 1. Create New Loader

```python
# src/preprocessing/loaders/custom_loader.py
from ..base import BaseLoader

class CustomLoader(BaseLoader):
    def load(self, config: dict):
        # Your loading logic
        for row in data:
            yield row
```

### 2. Create New Formatter

```python
# src/preprocessing/formatters/custom_formatter.py
from ..base import BaseFormatter

class CustomFormatter(BaseFormatter):
    def format(self, example: dict, config: dict) -> dict:
        # Your formatting logic
        return {"custom": "format"}
```

## 📋 Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `src/preprocessing/` module structure
- [ ] Implement base classes (`base.py`)
- [ ] Implement CSV, Excel, and HF loaders
- [ ] Create config schema for preprocessing

### Phase 2: Processing & Formatting
- [ ] Implement column filter processor
- [ ] Implement instruction builder
- [ ] Add schema context injector
- [ ] Create Alpaca and ChatML formatters
- [ ] Add validation layer

### Phase 3: Integration
- [ ] Update `main.py` with `prepare` command
- [ ] Add preprocessing section to `config.yaml`
- [ ] Write unit tests
- [ ] Update README.md

## 🧪 Example: Converting HF Dataset

```python
from src.preprocessing import DatasetPreprocessor

config = {
    "source": {
        "type": "hf",
        "path": "xlangai/spider",
        "split": "train"
    },
    "columns": {
        "ignore": ["db_id"],
        "input": ["question"],
        "output": ["query"],
        "system_prompt": "You are a Text-to-SQL expert."
    },
    "transformations": {
        "add_context": True,
        "context_columns": ["schema"]
    },
    "output": {
        "format": "instruction_output",
        "path": "data/train_sft.jsonl"
    }
}

preprocessor = DatasetPreprocessor(config)
preprocessor.run()
```

## 📚 Requirements

- Python 3.10+
- pandas (for CSV/Excel)
- datasets (for HuggingFace datasets)
- pyyaml (for config parsing)
- openpyxl (for Excel files)

## 🤝 Contributing

1. Create a new loader/processor/formatter following base class interface
2. Add tests in `tests/` directory
3. Update documentation
4. Submit pull request
