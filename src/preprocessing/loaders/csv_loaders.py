import csv
import os
import logging
from typing import Any, Dict, Generator, List, Optional

from ..base import BaseLoader, LoaderRegistry

logger = logging.getLogger(__name__)

@LoaderRegistry.register("csv")
class CsvLoader(BaseLoader): # child class of BaseLoader
    def __init__(self, config: Dict[str, Any]): # python contructor
        self.path = config.get("path", "")
        self.delimeter = config.get("delimeter", ",")
        self.encoding = config.get("encoding", "utf-8")

        ## BaseLoader needs these for it's internal logging
        self.dataset_name = self.path
        self.split = "train"

        super().__init__(config) ## getting all the attributes from baseloader so that we don't have repeat the same init again from baseloader
        
        self._validate_loader_config(self.path, "train")
    
    def _validate_loader_config(self, dataset_name: str, split: str):
        # for csv loader validation we need to these 4 validation
        ## 1. Presence validation(path exists)
        if not self.path:
            raise ValueError(f"File can't be Empty for Csv loader")
        ## 2. File existance check
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Unable to find the file.")
        ## 3. Delimiter check python only allows one delimiter
        if len(self.delimeter) != 1:
            raise ValueError("Delimiter must be exactly 1 chars.")
        ## 4. File type validation
        if not self.path.lower().endswith(".csv"):
            logging.warning(f"[bold yellow]{self.path} is not csv file.[/bold yellow]")

    def get_columns(self) -> List[str]:
        if not self._columns:
            logger.debug(f"Attempting to read columns from: {self.path}")
            # 1. open file
            with open(self.path, "r", encoding=self.encoding) as f:
                reader = csv.reader(f, delimiter=self.delimeter)
                try:
                    columns = next(reader)
                    self._columns = columns
                    logger.debug(f"Successfully extracted columns: {self._columns}")
                except StopIteration:
                    logger.error(f"Failed to read columns: File {self.path} is empty")
                    self._columns = []
        logging.info(f"List of Columns are: {self._columns}")
        return self._columns

    def get_row_count(self) -> Optional[int]:
        if not self._row_count:
            logger.debug(f"Calculating row count for: {self.path}")
            with open(self.path, 'r', encoding=self.encoding) as f:
                self._row_count = sum(1 for _ in f) - 1
            logger.debug(f"Row count calculation finished: {self._row_count} rows found")
        logging.info(f"Total Count of rows: {self._row_count}")
        return self._row_count

    def load(self) -> Generator[Dict[str, Any], None, None]:
        self._log_loading_start()
        logger.debug(f"Opening CSV file for streaming: {self.path} (Encoding: {self.encoding}, Delimiter: {self.delimeter})")

        with open(self.path, 'r', encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=self.delimeter)

            count = 0
            for i, row in enumerate(reader, start=1):
                try:
                    if i % 100 == 0:
                        logger.debug(f"Loader streaming progress: {i} rows read...")
                    yield row
                    count += 1
                except Exception as e:
                    logger.error(f"Error at row {i}: {str(e)}")
                    self._handle_row_error(i, e)

        logger.debug(f"Finished streaming from {self.path}. Total rows yielded: {count}")
        self._log_loading_complete(count)
