"""
This will act like a orchestrator for our preprocessing pipeline.
it will work in the 3 steps:
    1. Load the data (BaseLoader)
    2. Preprocess the data (BaseProcessor)
    3. Save the data (BaseFormatter)

"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generator, Callable
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Raised when loader configuration is invalid."""
    pass

class InvalidRowError(Exception):
    """Raised when a data row fails validation in strict mode."""
    pass

class LoaderType(Enum):
    """Enum for supported loader types."""
    CSV = "csv"
    EXCEL = "excel"
    HF = "hf"

@dataclass
class LoaderConfig:
    """
    Common configuration for all loaders.
    
    All loaders accept these core config options plus loader-specific options.
    """
    # Core required fields
    source_type: str                    # csv, excel, or hf
    path: str                          # File path or HF dataset name
    
    # Optional fields with defaults
    columns: Dict[str, List[str]] = field(default_factory=dict)
    ignore_columns: List[str] = field(default_factory=list)
    strict_mode: bool = False          # False = lenient error handling
    streaming: bool = True             # Always use generators
    limit: Optional[int] = None        # Max examples to process
    shuffle: bool = False              # Shuffle data
    seed: int = 42                     # Random seed for shuffle
    
    # Loader-specific config passed through
    extra_config: Dict[str, Any] = field(default_factory=dict)

class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    Design Principles:
    - Streaming-first: Always yield examples via generators
    - Lenient error handling: Log warnings, skip bad rows
    - Schema validation with warnings: Continue if columns missing
    
    Features:
    - Automatic progress tracking with tqdm
    - Sample extraction for testing
    - Column listing and validation
    - Dataset info retrieval
    - Shuffle capability
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loader with configuration.
        
        Args:
            config: Dictionary containing loader configuration
        """
        logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        self.config = config
        self._columns: List[str] = []
        self._row_count: Optional[int] = None
        self._is_shuffled: bool = False
        
        # Validate core config
        logger.debug("Validating core configuration...")
        self._validate_core_config()
        
        # Apply loader-specific validation
        logger.debug(f"Calling loader-specific validation for {self.__class__.__name__}")
        self._validate_loader_config(self.dataset_name, self.split)
        logger.debug(f"{self.__class__.__name__} initialization complete.")
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ CONFIGURATION VALIDATION                                         │
    # └─────────────────────────────────────────────────────────────────┘
    
    def _validate_core_config(self) -> None:
        """Validate core configuration fields."""
        required_fields = ["source_type", "path"]
        
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Validation failed: Missing required field '{field}'")
                raise ConfigValidationError(f"Missing required field: {field}")
        
        # Validate source_type
        valid_types = [t.value for t in LoaderType]
        if self.config["source_type"] not in valid_types:
            logger.error(f"Validation failed: Invalid source_type '{self.config['source_type']}'")
            raise ConfigValidationError(
                f"Invalid source_type: {self.config['source_type']}. "
                f"Must be one of: {valid_types}"
            )
        logger.debug("Core configuration validation successful.")
    
    @abstractmethod
    def _validate_loader_config(self, dataset_name: str, split: str) -> None:
        """
        Validate loader-specific configuration.
        Implement in subclass.
        """
        pass
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ CORE LOADING INTERFACE                                          │
    # └─────────────────────────────────────────────────────────────────┘
    
    @abstractmethod
    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Load dataset as a stream of examples.
        
        This is the main entry point for data loading.
        All subclasses MUST implement this method.
        
        Yields:
            Individual example dictionaries
            
        Error Handling:
            - Invalid rows are skipped with warnings
            - Processing continues even if some rows fail
        """
        pass
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ COLUMN OPERATIONS                                                │
    # └─────────────────────────────────────────────────────────────────┘
    
    @abstractmethod
    def get_columns(self) -> List[str]:
        """
        Get list of available column names.
        
        Returns:
            List of column names in the dataset
        """
        pass
    
    @abstractmethod
    def get_row_count(self) -> Optional[int]:
        """
        Get total number of rows.
        
        Returns:
            Total row count or None if unknown (streaming mode)
        """
        pass
    
    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate that required columns exist with warnings.
        
        Args:
            required_columns: List of column names that must exist
            
        Returns:
            True if all columns found, False otherwise
        """
        available = set(self.get_columns())
        missing = [col for col in required_columns if col not in available]
        
        if missing:
            warnings.warn(
                f"LoaderWarning: Missing columns: {missing}\n"
                f"Available columns: {available}",
                UserWarning
            )
            return False
        
        return True
    
    def select_columns(self, columns: List[str]) -> List[str]:
        """
        Return only the specified columns from available columns.
        
        Args:
            columns: List of column names to select
            
        Returns:
            Filtered list of existing columns
        """
        available = set(self.get_columns())
        return [col for col in columns if col in available]
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ SAMPLING & INSPECTION                                           │
    # └─────────────────────────────────────────────────────────────────┘
    
    def sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get first N examples for testing/inspection.
        
        Args:
            n: Number of examples to return
            
        Returns:
            List of example dictionaries
        """
        samples = []
        for i, example in enumerate(self.load()):
            if i >= n:
                break
            samples.append(example)
        return samples
    
    def info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with full dataset metadata
        """
        return {
            "loader_type": self.__class__.__name__,
            "source_type": self.config.get("source_type"),
            "path": self.config.get("path"),
            "columns": self.get_columns(),
            "row_count": self.get_row_count(),
            "streaming": self.config.get("streaming", True),
            "shuffle": self.config.get("shuffle", False),
            "config": self.config
        }
    
    def preview(self, n: int = 3) -> None:
        """
        Print a preview of the dataset to console.
        
        Args:
            n: Number of examples to show
        """
        info = self.info()
        print(f"\n{'='*60}")
        print(f"Dataset Preview: {info['loader_type']}")
        print(f"{'='*60}")
        print(f"Source: {info['path']}")
        print(f"Columns ({len(info['columns'])}): {info['columns']}")
        print(f"Rows: {info['row_count'] or 'Unknown (streaming)'}")
        print(f"Streaming: {info['streaming']}")
        print(f"\nSample Examples:")
        print(f"{'-'*60}")
        
        for i, sample in enumerate(self.sample(n)):
            print(f"\nExample {i+1}:")
            for key, value in sample.items():
                # Truncate long values
                val_str = str(value)[:100]
                if len(str(value)) > 100:
                    val_str += "..."
                print(f"  {key}: {val_str}")
        print(f"{'='*60}\n")
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ PROGRESS TRACKING                                               │
    # └─────────────────────────────────────────────────────────────────┘
    
    def _get_progress_bar(self, desc: str = "Loading", 
                          total: Optional[int] = None):
        """
        Get tqdm progress bar instance.
        
        Args:
            desc: Description for progress bar
            total: Total items (None for unknown)
            
        Returns:
            tqdm object or None if not available
        """
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, unit="rows")
        except ImportError:
            warnings.warn("tqdm not installed. Progress tracking disabled.")
            return None
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ ERROR HANDLING                                                  │
    # └─────────────────────────────────────────────────────────────────┘
    
    def _handle_row_error(self, row_num: int, 
                          error: Exception,
                          context: str = "processing") -> None:
        """
        Handle invalid row with lenient error handling.
        
        Args:
            row_num: Current row number
            error: The exception that occurred
            context: Description of where error occurred
        """
        if self.config.get("strict_mode", False):
            # Strict mode: re-raise the exception
            raise InvalidRowError(
                f"Error {context} row {row_num}: {str(error)}"
            ) from error
        else:
            # Lenient mode: log warning and continue
            warnings.warn(
                f"Skipping row {row_num} due to error during {context}: "
                f"{str(error)}",
                UserWarning
            )
    
    def _log_loading_start(self) -> None:
        """Log loading operation start."""
        logger.info(
            f"Starting data load: {self.__class__.__name__} "
            f"from {self.config.get('path')}"
        )
    
    def _log_loading_complete(self, total_processed: int) -> None:
        """Log loading operation completion."""
        logger.info(
            f"Data load complete. Processed {total_processed} examples."
        )
    
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ ITERATOR WRAPPERS                                               │
    # └─────────────────────────────────────────────────────────────────┘
    
    def stream_with_progress(self, desc: str = "Processing") -> Generator:
        """
        Wrap load() with progress tracking.
        
        Args:
            desc: Progress bar description
            
        Returns:
            Generator with progress bar
        """
        total = self.get_row_count()
        progress = self._get_progress_bar(desc, total)
        
        for example in self.load():
            yield example
            if progress:
                progress.update(1)
        
        if progress:
            progress.close()
    
    def stream_with_shuffle(self, 
                           buffer_size: int = 1000) -> Generator:
        """
        Wrap load() with shuffling using a buffer.
        
        Args:
            buffer_size: Number of examples to buffer for shuffling
            
        Returns:
            Shuffled generator
        """
        import random
        
        buffer = []
        for example in self.load():
            if len(buffer) < buffer_size:
                buffer.append(example)
            else:
                # With probability buffer_size/(buffer_size+1), replace
                # This creates a streaming shuffle effect
                idx = random.randint(0, buffer_size)
                if idx < buffer_size:
                    yield buffer[idx]
                    buffer[idx] = example
                else:
                    yield example
        
        # Yield remaining buffer items
        random.shuffle(buffer)
        for item in buffer:
            yield item
    
    def stream_with_limit(self, limit: int) -> Generator:
        """
        Wrap load() with a maximum limit.
        
        Args:
            limit: Maximum number of examples to yield
            
        Returns:
            Limited generator
        """
        count = 0
        for example in self.load():
            if count >= limit:
                break
            yield example
            count += 1 

class LoaderRegistry:
    """
    Registry for loader plugins.
    Enables automatic discovery and instantiation.
    """
    
    _loaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a loader class."""
        def decorator(loader_class):
            cls._loaders[name] = loader_class
            return loader_class
        return decorator
    
    @classmethod
    def get_loader_class(cls, loader_type: str) -> type:
        """Get loader class by type string."""
        if loader_type not in cls._loaders:
            raise ValueError(
                f"Unknown loader type: {loader_type}. "
                f"Available: {list(cls._loaders.keys())}"
            )
        return cls._loaders[loader_type]
    
    @classmethod
    def create_loader(cls, config: Dict[str, Any]):
        """
        Create loader instance from config.
        
        Args:
            config: Loader configuration dictionary
            
        Returns:
            Instantiated loader object
        """
        loader_type = config.get("source_type")
        logger.debug(f"Requesting loader creation for type: {loader_type}")
        loader_class = cls.get_loader_class(loader_type)
        logger.debug(f"Found loader class: {loader_class.__name__}")
        return loader_class(config)


def create_loader(config: Dict[str, Any]) -> BaseLoader:
    """
    Factory function to create a loader from config.
    
    Args:
        config: Loader configuration dictionary
        
    Returns:
        Instantiated loader object
    """
    loader_type = config.get("source_type")
    loader_class = LoaderRegistry.create_loader(config)
    return loader_class

class BaseProcessor(ABC):
    """
    Abstract base class for data processors.
    Handles mapping, filtering, and cleaning.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def process(self, data_stream: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        """Process a stream of data."""
        pass

class BaseFormatter(ABC):
    """
    Abstract base class for data formatters.
    Handles conversion to training formats (ChatML, JSONL, etc).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def format(self, data_stream: Generator[Dict[str, Any], None, None], output_path: str) -> None:
        """Format and save the data stream."""
        pass
