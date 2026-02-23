"""
Goal: Connect to various data sources and normalize them into a standard intermediate format.
"""
import logging
import sqlalchemy as db
import pandas as pd

# Basic logging setup
logging.basicConfig(level=logging.INFO)

class ProcessDatabase:
    def __init__(self, connectivity_uri, table_name, target_col=None, ignore_col=None):
        self.connectivity_uri = connectivity_uri
        self.table_name = table_name
        self.target_col = target_col
        self.ignore_col = ignore_col
        self.engine, self.columns = self.ingest_database()

    def ingest_database(self):
        # Disable echo for cleaner output
        engine = db.create_engine(self.connectivity_uri, echo=False)
        try:
            with engine.connect() as conn:
                # Use text() for SQLAlchemy 2.0 compatibility
                result = conn.execute(db.text(f"SELECT * FROM {self.table_name} LIMIT 1"))
                columns = list(result.keys())
            logging.info(f"Connected to {self.connectivity_uri}, table: {self.table_name}")
            return engine, columns
        except Exception as e:
            raise RuntimeError(f"Unable to connect with the database: {e}")

    def fetch_filtered_data(self):
        try:
            # Recommended for SQLAlchemy 2.0 + Pandas: Use a connection context
            with self.engine.connect() as conn:
                # Use read_sql_query with sqlalchemy.text for string queries
                query = db.text(f"SELECT * FROM {self.table_name}")
                table = pd.read_sql_query(query, con=conn)

            if self.ignore_col:
                ignore_list = [self.ignore_col] if isinstance(self.ignore_col, str) else self.ignore_col
                table = table.drop(columns=ignore_list, errors="ignore")

            return table
        except Exception as e:
            raise RuntimeError(f"Unable to load the table: {e}")

if __name__ == "__main__":
    # Test with a local database if it exists, otherwise use a memory one for demonstration
    import os
    db_uri = "sqlite:///chinook.db"
    
    # Create a dummy database if chinook.db doesn't exist for the test
    if not os.path.exists("chinook.db"):
        print("Creating dummy chinook.db for testing...")
        engine = db.create_engine(db_uri)
        df = pd.DataFrame({'AlbumId': [1], 'Title': ['Test Album'], 'ArtistId': [1]})
        df.to_sql('albums', engine, index=False)

    processor = ProcessDatabase(db_uri, "albums")
    print("Columns:", processor.columns)
    df = processor.fetch_filtered_data()
    print("\nData Head:")
    print(df.head())
