from src.preprocessing import create_loader
import logging

# Setup logging to see the info messages
logging.basicConfig(level=logging.INFO)

def test_titanic_csv():
    config = {
        "source_type": "csv",
        "path": "data/Titanic-Dataset.csv",
        "delimeter": ",",
        "encoding": "utf-8"
    }
    
    # 1. Create the loader
    print("\n--- Initializing Loader ---")
    loader = create_loader(config)
    
    # 2. Test get_columns
    print("\n--- Testing get_columns ---")
    cols = loader.get_columns()
    print(f"Columns: {cols}")
    
    # 3. Test get_row_count
    print("\n--- Testing get_row_count ---")
    count = loader.get_row_count()
    print(f"Total Rows: {count}")
    
    # 4. Test preview (uses sample/load)
    print("\n--- Testing Preview (First 3 rows) ---")
    loader.preview(3)
    
    # 5. Test streaming load
    print("\n--- Testing Streaming Load (First 2 rows) ---")
    for i, row in enumerate(loader.load()):
        if i >= 2:
            break
        print(f"Row {i+1}: {row}")

if __name__ == "__main__":
    test_titanic_csv()
