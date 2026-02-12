"""
HuggingFace Dataset Loader
Loads datasets from Hugging Face Hub or local files.
Supports streaming, split handling, and automatic schema detection.
"""
import logging
import warnings
from typing import Any, Dict, Generator, List, Optional, Union
from datasets import (
    load_dataset,
    IterableDatasetDict,
    Dataset,
    DatasetDict,
)
from ..base import BaseLoader, LoaderRegistry
logger = logging.getLogger(__name__)

@LoaderRegistry.register("hf")
class HfLoader(BaseLoader):
    """
    Loader for Hugging Face datasets.
    
    Features:
    - Load remote HF datasets by name
    - Load local JSON/JSONL/CSV files
    - Split support (train, validation, test)
    - Streaming mode for large datasets
    - Automatic column detection
    - Config-based filtering
    
    Example Config:
        {
            "source_type": "hf",
            "path": "xlangai/spider",
            "split": "train",
            "streaming": True,
            "columns": {
                "input": ["question"],
                "output": ["query"]
            }
        }
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HuggingFace loader.
        
        Args:
            config: Configuration dictionary with the following options:
                path (str): HF dataset name OR local file path
                split (str): Dataset split to load (default: "train")
                name (str): Optional dataset configuration name
                streaming (bool): Use streaming mode (default: True)
                trust_remote_code (bool): Trust remote code (default: False)
                columns (dict): Column mapping for input/output
        """
        # ========== SET ALL ATTRIBUTES FIRST ==========
        self.dataset_name = config.get("path", "")
        self.split = config.get("split", "train")
        self.dataset_config = config.get("name", None)
        self.trust_remote_code = config.get("trust_remote_code", False)
        self._dataset = None
        self._is_streaming = config.get("streaming", True)
        
        # ========== NOW SAFE TO VALIDATE ==========
        self._validate_loader_config(self.dataset_name, self.split)
        
        # ========== CALL PARENT LAST ==========
        super().__init__(config)

    def _validate_loader_config(self, dataset_name: str, split: str) -> None:
        """
        Validate HuggingFace-specific configuration.
        
        Checks:
        - Dataset path is not empty
        - Split is valid (if provided)
        """
        if not self.dataset_name:
            raise ValueError("HF dataset 'path' cannot be empty")
    
    def _load_hf_dataset(self) -> Union[Dataset, DatasetDict, IterableDatasetDict]:
        """
        Load dataset from Hugging Face.
        
        Returns:
            Loaded dataset object
            
        Raises:
            ValueError: If dataset cannot be loaded
        """
        if self._dataset is not None:
            return self._dataset
        
        logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
        
        try:
            # Check if it's a local file
            if self._is_local_path(self.dataset_name):
                dataset = self._load_local_file()
            else:
                # Load from HF Hub
                dataset = load_dataset(
                    self.dataset_name,
                    name=self.dataset_config,
                    trust_remote_code=self.trust_remote_code,
                )
            
            logger.info(
                f"Successfully loaded dataset: {self.dataset_name}. "
                f"Available splits: {list(dataset.keys())}"
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load HF dataset: {str(e)}")
            raise ValueError(
                f"Could not load HuggingFace dataset '{self.dataset_name}': {str(e)}"
            ) from e

    def _is_local_path(self, path: str) -> bool:
        """
        Check if the path is a local file.
        
        Args:
            path: Path to check
            
        Returns:
            True if local file, False otherwise
        """
        # Check for absolute paths starting with /
        if path.startswith("/"):
            return True
        
        # Check for relative paths
        import os
        if os.path.exists(path):
            return True
        
        # Check for common file extensions
        local_extensions = [".json", ".jsonl", ".csv", ".parquet", ".txt"]
        return any(path.endswith(ext) for ext in local_extensions)
    
    def _load_local_file(self) -> Union[Dataset, DatasetDict]:
        """
        Load a local file as a HuggingFace dataset.
        
        Detects format from file extension and loads accordingly.
        
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If file format is not supported
        """
        import os
        
        path = self.dataset_name
        
        if not os.path.exists(path):
            raise ValueError(f"Local file not found: {path}")
        
        logger.info(f"Loading local file: {path}")
        
        # Detect format from extension
        if path.endswith(".json") or path.endswith(".jsonl"):
            return load_dataset("json", data_files=path, split=self.split)
        
        elif path.endswith(".csv"):
            return load_dataset("csv", data_files=path, split=self.split)
        
        elif path.endswith(".parquet"):
            return load_dataset("parquet", data_files=path, split=self.split)
        
        else:
            # Try to detect automatically
            logger.warning(
                f"Unknown file extension for: {path}. "
                f"Attempting to load as JSON by default."
            )
            return load_dataset("json", data_files=path, split=self.split)

    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Load HF dataset as streaming generator.
        
        This is the main entry point for data loading.
        Yields individual examples from the dataset.
        
        Yields:
            Dict[str, Any]: Individual dataset examples
            
        Raises:
            DatasetLoadingError: If dataset cannot be loaded
            
        Example:
            >>> loader = HfLoader({"path": "spider", "split": "train"})
            >>> for example in loader.load():
            ...     print(example)
        """
        self._log_loading_start()
        
        dataset = self._load_hf_dataset()
        
        # Get the specific split
        split_dataset = self._get_split(dataset, self.split)
        
        # Determine iteration mode
        if self._is_streaming:
            iterator = self._iterate_streaming(split_dataset)
        else:
            iterator = self._iterate_regular(split_dataset)
        
        # Track progress
        total_processed = 0
        progress = self._get_progress_bar(
            desc=f"Loading HF dataset: {self.dataset_name}"
        )
        
        if progress is not None and hasattr(progress, 'update'):
            progress.update(1)
        
        try:
            for example in iterator:
                yield example
                total_processed += 1
                if progress is not None and hasattr(progress, 'update'):
                    progress.update(1)
            
            self._log_loading_complete(total_processed)
            
        finally:
            if progress is not None and hasattr(progress, 'close'):
                progress.close()

    def _get_split(
        self, 
        dataset: Union[Dataset, DatasetDict, IterableDatasetDict],
        split: str
    ) -> Any:
        """
        Extract a specific split from the dataset.
        
        Args:
            dataset: Full dataset object
            split: Name of split to extract
            
        Returns:
            Split dataset
            
        Raises:
            ValueError: If split doesn't exist
        """
        # Handle DatasetDict (most common case)
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(
                    f"Split '{split}' not found. "
                    f"Available splits: {available_splits}. "
                    f"Using first available split."
                )
                split = available_splits[0]
            return dataset[split]
        
        # Handle IterableDatasetDict
        if isinstance(dataset, IterableDatasetDict):
            if split not in dataset:
                available = list(dataset.keys())
                split = available[0] if available else "train"
                logger.info(f"Using split: {split}")
            return dataset[split]
        
        # Handle single Dataset (already a split)
        return dataset
    
    def _iterate_streaming(self, dataset: Any) -> Generator[Dict, None, None]:
        """
        Iterate over a streaming dataset.
        
        Args:
            dataset: Streaming-compatible dataset
            
        Yields:
            Individual examples
        """
        # Enable streaming if not already enabled
        if hasattr(dataset, 'streaming') and not dataset.streaming:
            dataset = dataset.streaming()
        
        for example in dataset:
            yield example
    
    def _iterate_regular(self, dataset: Any) -> Generator[Dict, None, None]:
        """
        Iterate over a regular (non-streaming) dataset.
        
        Args:
            dataset: Regular dataset
            
        Yields:
            Individual examples
        """
        for example in dataset:
            yield example
    
    def get_columns(self) -> List[str]:
        """
        Get list of available column names.
        
        Returns:
            List of column names in the dataset
        """
        if self._columns:
            return self._columns
        
        try:
            dataset = self._load_hf_dataset()
            split = self._get_split(dataset, self.split)
            
            if hasattr(split, 'features'):
                self._columns = list(split.features.keys())
            elif hasattr(split, 'column_names'):
                self._columns = split.column_names
            else:
                # For streaming, get from first example
                for example in self._iterate_streaming(split):
                    self._columns = list(example.keys())
                    break
            
            logger.debug(f"Detected columns: {self._columns}")
            return self._columns
            
        except Exception as e:
            logger.warning(f"Could not detect columns: {str(e)}")
            return []
    
    def get_row_count(self) -> Optional[int]:
        """
        Get total number of rows.
        
        Note: Returns None for streaming datasets.
        
        Returns:
            Total row count or None if unknown
        """
        if self._is_streaming:
            logger.debug("Row count unavailable in streaming mode")
            return None
        
        try:
            dataset = self._load_hf_dataset()
            split = self._get_split(dataset, self.split)
            self._row_count = len(split)
            return self._row_count
            
        except Exception as e:
            logger.warning(f"Could not get row count: {str(e)}")
            return None
    
    def get_split_names(self) -> List[str]:
        """
        Get list of available split names.
        
        Returns:
            List of split names
        """
        try:
            dataset = self._load_hf_dataset()
            if isinstance(dataset, DatasetDict):
                return list(dataset.keys())
            return [self.split]
        except Exception as e:
            logger.warning(f"Could not get split names: {str(e)}")
            return []
    
    def info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Overrides base method to add HF-specific info.
        
        Returns:
            Dictionary with full dataset metadata
        """
        base_info = super().info()
        
        hf_info = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "available_splits": self.get_split_names(),
            "streaming": self._is_streaming,
            "trust_remote_code": self.trust_remote_code,
        }
        
        return {**base_info, **hf_info}
    
    def preview(self, n: int = 3) -> None:
        """
        Print a preview of the dataset to console.
        
        Args:
            n: Number of examples to show
        """
        info = self.info()
        
        print(f"\n{'='*70}")
        print(f"HuggingFace Dataset Preview")
        print(f"{'='*70}")
        print(f"Dataset Name: {info['dataset_name']}")
        print(f"Split: {info['split']}")
        print(f"Available Splits: {info['available_splits']}")
        print(f"Columns ({len(info['columns'])}): {info['columns']}")
        print(f"Rows: {info['row_count'] or 'Unknown (streaming mode)'}")
        print(f"Streaming: {info['streaming']}")
        print(f"\nSample Examples:")
        print(f"{'-'*70}")
        
        for i, sample in enumerate(self.sample(n)):
            print(f"\nExample {i+1}:")
            for key, value in sample.items():
                val_str = str(value)[:80]
                if len(str(value)) > 80:
                    val_str += "..."
                print(f"  {key}: {val_str}")
        
        print(f"{'='*70}\n")
    
    def filter_by_column(self, column: str, values: List[Any]) -> "HfLoader":
        """
        Create a new loader that filters examples by column values.
        
        Args:
            column: Column name to filter on
            values: List of values to include
            
        Returns:
            New HfLoader instance with filtering enabled
            
        Example:
            >>> loader = HfLoader({"path": "spider", "split": "train"})
            >>> filtered = loader.filter_by_column("difficulty", ["easy", "medium"])
        """
        import copy
        
        # Create a copy of config with filter
        new_config = copy.deepcopy(self.config)
        new_config["filter_column"] = column
        new_config["filter_values"] = values
        
        return HfLoader(new_config)
    
    def select_columns(self, columns: List[str]) -> "HfLoader":
        """
        Create a new loader that only loads specified columns.
        
        Args:
            columns: List of columns to include
            
        Returns:
            New HfLoader instance with column selection
            
        Example:
            >>> loader = HfLoader({"path": "spider"})
            >>> slim = loader.select_columns(["question", "query"])
        """
        import copy
        
        new_config = copy.deepcopy(self.config)
        new_config["columns"] = columns
        
        return HfLoader(new_config)  


# if __name__ == "__main__":
#     loader = HfLoader({
#         "source_type": "hf",
#         "path": "xlangai/spider",
#         "split": "train",
#         "streaming": True,
#         "columns": {
#             "input": ["question"],
#             "output": ["query"]
#         }
#     })
#     for example in loader._load_hf_dataset():
#         print(example)