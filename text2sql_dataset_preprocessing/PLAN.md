# Text2SQL Data Preprocessing Module - Production Roadmap

**Objective:** Build a robust, scalable, and modular pipeline to gather, clean, and format Text-to-SQL datasets (Spider, BIRD, Gretel, etc.) for Supervised Fine-Tuning (SFT).

**Current Status:** Prototype (MVP) with basic functionality but lacking robustness, modularity, and comprehensive error handling.

---

## 🚀 Phase 1: Immediate Fixes & MVP Stabilization
*Goal: Get the current pipeline running end-to-end without errors to establish a baseline.*

- [ ] **Fix Syntax Errors**: 
    - [ ] Resolve `SyntaxError` in `src/schema_parser.py` (line 34).
    - [ ] Verify `src/formatter.py` fixes are persistent.
- [ ] **Complete `main.py` Integration**: 
    - [ ] Ensure `process_datasets` calls `load_spider_schemas` correctly.
    - [ ] Verify argument passing between `gatherer`, `cleaner`, and `formatter`.
- [ ] **Basic Output Verification**:
    - [ ] Run the pipeline and inspect `data/train_sft.jsonl`.
    - [ ] Check `data/raw/spider_tables.json` parsing correctness.

## 🏗️ Phase 2: Refactoring for Modularity (The "Right Way")
*Goal: Decouple logic so adding new datasets (e.g., WikiSQL) is strictly additive (Open/Closed Principle).*

- [ ] **Define Abstract Base Classes**:
    - [ ] Create `DatasetHandler` interface in `src/interfaces.py`.
        - Methods: `download()`, `clean(entry)`, `format(entry)`.
- [ ] **Implement Concrete Handlers**:
    - [ ] `SpiderHandler` (encapsulate `spider_tables.json` logic here).
    - [ ] `BirdHandler` (handle `evidence` fields).
    - [ ] `GretelHandler` (handle `sql_context`).
- [ ] **Factory Pattern**:
    - [ ] Create `DatasetHandlerFactory` to instantiate handlers based on config/dataset name.
- [ ] **Refactor `main.py`**:
    - [ ] Remove if/else logic for dataset types.
    - [ ] Iterate through configured datasets and delegate to their specific handlers.

## 🛡️ Phase 3: Robustness & Data Integrity
*Goal: Ensure the pipeline handles bad data gracefully and provides visibility.*

- [ ] **Strict Typing & Validation**:
    - [ ] Define Pydantic models for `RawEntry` and `SFTEntry` in `src/schemas.py`.
    - [ ] Enforce schema validation during the `clean` step.
- [ ] **Advanced Error Handling**:
    - [ ] Create custom exceptions (e.g., `SchemaNotFoundError`, `InvalidSQLError`).
    - [ ] Implement a "Dead Letter Queue" mechanism (save failed entries to `errors.jsonl` instead of just logging).
- [ ] **Structured Logging**:
    - [ ] Configure `logging` to output JSON (better for parsing logs).
    - [ ] Include context (dataset name, line number, error details) in logs.

## ⚡ Phase 4: Scalability & Performance
*Goal: Optimize for processing millions of records.*

- [ ] **Parallel Processing**:
    - [ ] Replace single-threaded loop with `multiprocessing.Pool` or `concurrent.futures`.
    - [ ] Process files in chunks/batches.
- [ ] **Configuration Management**:
    - [ ] Replace raw `yaml` loading with a typed config class (e.g., using `pydantic-settings`).
    - [ ] Allow overriding config via environment variables.

## 🧪 Phase 5: Quality Assurance
*Goal: Guarantee reliability through automated testing.*

- [ ] **Unit Tests**:
    - [ ] Test `cleaner` with edge cases (empty strings, malicious SQL).
    - [ ] Test `formatter` for exact SFT template matching.
    - [ ] Test `schema_parser` with mock JSON data.
- [ ] **Integration Tests**:
    - [ ] Run a mini-pipeline on a small sample dataset (10 rows) and assert output file exists and is valid JSONL.
- [ ] **Linting & formatting**:
    - [ ] Set up `ruff` and `mypy` configuration.
    - [ ] Enforce type hints across the codebase.

---

## 📂 Source of Truth
This document will serve as the master plan. Updates to the architecture or new requirements should be reflected here first.
