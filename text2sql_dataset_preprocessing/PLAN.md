# Text2SQL Data Preprocessing Module - Production Roadmap

**Objective:** Build a robust, scalable, and modular pipeline to gather, clean, and format Text-to-SQL datasets for Supervised Fine-Tuning (SFT).

---

## ✅ Phase 1: Immediate Fixes & MVP Stabilization
- [x] **Fix Syntax Errors**: Multi-line f-string issues in `formatter.py` resolved.
- [x] **Complete `main.py` Integration**: Seamless flow between gatherer, cleaner, and generator.
- [x] **Basic Output Verification**: `train_sft.jsonl` verified with actual schemas.

## ✅ Phase 2: Refactoring for Modularity
- [x] **Define Logic Separation**: Decoupled Cleaner from Formatter and Publisher.
- [x] **Implement Schema Fallback**: Added `schema_generator.py` to handle missing metadata files.
- [x] **Automation**: Added `poethepoet` tasks for standardized runs.

## ✅ Phase 3: Robustness & Data Integrity
- [x] **Strict SQL Validation**: Integrated `sqlglot` for syntax checking.
- [x] **Versioning**: Implemented HF Tagging for data versioning.
- [x] **Security**: Secrets and large data removed from Git history.

## 🚀 Phase 4: Future Elevations (MLOps Next Steps)
- [ ] **Parallel Processing**: Utilize `multiprocessing` for cleaning millions of records.
- [ ] **Execution Accuracy Script**: Build a script to verify SQL results against real SQLite databases.
- [ ] **Value Anchoring**: Enhance `schema_generator` to include sample row values in prompts.
- [ ] **Full MLOps Tracking**: Complete MLflow integration for model registry.
