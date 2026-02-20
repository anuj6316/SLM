#!/usr/bin/env python3
"""
SLM Text-to-SQL Pipeline - Unified entry point.

Usage:
    python main.py data all           # Run full data pipeline
    python main.py train sft          # Train with SFT
    python main.py eval               # Evaluate model
    python main.py infer -i           # Interactive inference
    python main.py infer --trace      # Traced inference with MLflow
    python main.py pipeline full      # Run complete pipeline

For more options:
    python main.py --help
    python main.py data --help
    python main.py train --help
"""

from dotenv import load_dotenv

from slm.cli import main

if __name__ == "__main__":
    load_dotenv()
    main()
