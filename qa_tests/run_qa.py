import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.preprocessing import create_loader, SFTProcessor

# Configure logging to capture all details
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger("QA_Tester")

def run_test(name, config, expected_behavior):
    print(f"\n{'='*20} TEST: {name} {'='*20}")
    print(f"Goal: {expected_behavior}")
    try:
        loader = create_loader(config)
        print(f"SUCCESS: Loader initialized.")
        
        # Test get_columns
        cols = loader.get_columns()
        print(f"Columns found: {cols}")
        
        # Test get_row_count
        count = loader.get_row_count()
        print(f"Row count: {count}")
        
        # Test load
        rows = list(loader.load())
        print(f"Rows loaded: {len(rows)}")
        
        # Test Process
        processor = SFTProcessor(config)
        processed = list(processor.process(iter(rows)))
        print(f"Rows processed: {len(processed)}")
        
    except Exception as e:
        print(f"CAUGHT ERROR: {type(e).__name__}: {str(e)}")

# Test Cases
tests = [
    {
        "name": "Missing File",
        "config": {
            "source_type": "csv",
            "path": "qa_tests/data/non_existent.csv",
            "delimeter": ",",
            "columns": {"instruction": "col1", "output": "col2"}
        },
        "expected": "Should raise FileNotFoundError during initialization."
    },
    {
        "name": "Empty File",
        "config": {
            "source_type": "csv",
            "path": "qa_tests/data/empty.csv",
            "delimeter": ",",
            "columns": {"instruction": "col1", "output": "col2"}
        },
        "expected": "Should handle empty file gracefully."
    },
    {
        "name": "Wrong Delimiter",
        "config": {
            "source_type": "csv",
            "path": "qa_tests/data/wrong_delimiter.csv",
            "delimeter": ",", 
            "columns": {"instruction": "col1", "output": "col2"}
        },
        "expected": "Should load 1 column and 0 processed rows."
    },
    {
        "name": "Missing Mapping Columns",
        "config": {
            "source_type": "csv",
            "path": "qa_tests/data/only_headers.csv",
            "delimeter": ",",
            "columns": {"instruction": "wrong_col", "output": "col2"}
        },
        "expected": "Should load 0 processed rows."
    },
    {
        "name": "Malformed Rows",
        "config": {
            "source_type": "csv",
            "path": "qa_tests/data/malformed.csv",
            "delimeter": ",",
            "columns": {"instruction": "col1", "output": "col2"}
        },
        "expected": "Should skip or handle malformed rows."
    }
]

if __name__ == "__main__":
    for t in tests:
        run_test(t["name"], t["config"], t["expected"])
