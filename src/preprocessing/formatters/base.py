import json
import os
import logging
from typing import Any, Dict, Generator
from ..base import BaseFormatter

logger = logging.getLogger(__name__)

class JsonlFormatter(BaseFormatter):
    """
    Saves the data stream into a .jsonl file.
    Each line is a separate JSON object.
    """
    
    def format(self, data_stream: Generator[Dict[str, Any], None, None], output_path: str) -> None:
        """
        Consumes the stream and writes to disk.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        count = 0
        logger.info(f"Writing processed data to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data_stream:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
                
        logger.info(f"Finished writing {count} examples to {output_path}")
