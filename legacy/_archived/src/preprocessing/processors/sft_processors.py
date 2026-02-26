from typing import Any, Dict, Generator, List, Optional
import logging
from ..base import BaseProcessor

logger = logging.getLogger(__name__)

class SFTProcessor(BaseProcessor):
    """
    Standard processor for Text-to-SQL tasks.
    Handles column mapping from source schema to instruction/input/output format.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.column_mapping = config.get("columns", {})
        self.strict_mode = config.get("strict_mode", False)
        self._first_row_logged = False

    def process(self, data_stream: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        """
        Processes the data stream by mapping columns and applying basic cleaning.
        """
        logger.info("SFTProcessor: Starting data stream processing...")
        for i, example in enumerate(data_stream):
            try:
                processed = self._map_columns(example)
                if processed:
                    yield processed
            except Exception as e:
                logger.error(f"Processor encountered error at row {i}: {str(e)}")
                if self.strict_mode:
                    raise e
        logger.info("SFTProcessor: Processing complete.")

    def _map_columns(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Maps source keys to standard keys based on config.
        Standard keys: instruction, input, output
        """
        result = {}
        
        if not self._first_row_logged:
            logger.info("Verifying column mapping with first row sample...")
        
        for target_key in ["instruction", "input", "output"]:
            source = self.column_mapping.get(target_key)

            if isinstance(source, list):
                # combine multiple columns: "Col1: val1 | Col2: val2"
                parts = [f"{col}: {example.get(col, '')}" for col in source]
                result[target_key] = " | ".join(parts)
            elif source:
                # single column mapping
                val = example.get(source)
                result[target_key] = str(val) if val is not None else ""
            else:
                # fallback to empty string
                result[target_key] = ""
            
            if not self._first_row_logged:
                logger.debug(f"  [Mapping] {target_key} <- {source}")
        
        # validation: Usually SFT requires at least instruction and output
        if not result["instruction"] or not result["output"]:
            if not self._first_row_logged:
                logger.warning("First row rejected: missing required 'instruction' or 'output' data. Check your config mapping!")
            return None
        
        if not self._first_row_logged:
            logger.info("Column mapping verified successfully.")
            self._first_row_logged = True
            
        return result
