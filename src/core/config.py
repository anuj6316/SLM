import yaml
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AppConfig:
    """A wrapper class to allow dot-notation access to configuration."""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, AppConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from a YAML file and returns an AppConfig object."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default empty config.")
        return AppConfig({})
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
        return AppConfig(config_dict)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return AppConfig({})
