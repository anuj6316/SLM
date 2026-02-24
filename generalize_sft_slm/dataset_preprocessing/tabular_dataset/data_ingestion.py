"""
Goal: Connect to various data sources and normalize them into a standard intermediate format.
"""
import logging
import os
import re

import pandas as pd
import polars as pl
import sqlalchemy as db

from ..config import TabularDataset
from .utils import _format_for_unsloth

logger = logging.getLogger(__name__)

# Allowlist: table names must be simple identifiers (letters, digits, underscores, dots)
_TABLE_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')


class ProcessDatabase:
    def __init__(self, connectivity_uri: str, table_name: str, target_col: str = None, ignore_col=None):
        if not _TABLE_NAME_RE.match(table_name):
            raise ValueError(
                f"Invalid table name {table_name!r}. "
                "Only letters, digits, underscores, and dots are allowed."
            )
        self.connectivity_uri = connectivity_uri
        self.table_name = table_name
        self.target_col = target_col
        self.ignore_col = ignore_col
        self.engine, self.columns = self._ingest_database()

    def _ingest_database(self):
        engine = db.create_engine(self.connectivity_uri, echo=False)
        try:
            with engine.connect() as conn:
                # Use text() for SQLAlchemy 2.0 compatibility
                result = conn.execute(db.text(f"SELECT * FROM {self.table_name} LIMIT 1"))
                columns = list(result.keys())
            logger.info(f"Connected to database, table: {self.table_name}")
            return engine, columns
        except Exception as e:
            raise RuntimeError(f"Unable to connect with the database: {e}")

    def fetch_filtered_data(self) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                query = db.text(f"SELECT * FROM {self.table_name}")
                table = pd.read_sql_query(query, con=conn)

            if self.ignore_col:
                ignore_list = [self.ignore_col] if isinstance(self.ignore_col, str) else list(self.ignore_col)
                table = table.drop(columns=ignore_list, errors="ignore")

            return table
        except Exception as e:
            raise RuntimeError(f"Unable to load the table: {e}")


class ProcessLocalFile:
    def __init__(self, clf: TabularDataset):
        self.path = clf.path
        self.target_col = clf.target_col
        self.ignore_cols = clf.ignore_cols
        self.df, self.columns = self._load_df()

    def _load_df(self) -> tuple[pl.DataFrame, list[str]]:
        ext = os.path.splitext(self.path)[-1].lower()

        if ext == ".csv":
            df = pl.read_csv(self.path)
            return df, df.columns
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(self.path)
            return pl.from_pandas(df), list(df.columns)
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .csv, .xls, .xlsx")
