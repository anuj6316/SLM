from generalize_sft_slm.dataset_preprocessing.tabular_dataset.data_ingestion import ProcessDatabase
import os

db_path = "/home/mindmap/Desktop/SLM/data/raw/databases/academic/academic.sqlite"
table_name = "author"
connectivity_uri = f"sqlite:///{db_path}"

try:
    print(f"Testing ingestion from: {db_path} table: {table_name}")
    processor = ProcessDatabase(connectivity_uri, table_name)
    data = processor.fetch_filtered_data()
    print("Ingested Data (First 5 rows):")
    print(data.head())
    print("Columns available:")
    print(processor.columns)
except Exception as e:
    print(f"Error during test: {e}")
