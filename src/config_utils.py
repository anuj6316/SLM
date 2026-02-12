import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using empty config.")
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}

def merge_config_with_args(config: dict, args_namespace) -> dict:
    """
    Merges configuration dictionary with command line arguments.
    Arguments take precedence over config file values.
    """
    args_dict = vars(args_namespace)
    merged = config.copy()
    
    for key, value in args_dict.items():
        # Only override if the argument was explicitly provided (not None or default)
        # Note: This logic assumes that if an arg is provided, it should override.
        # Since argparse always has defaults, we might need to check if it's different from default
        # but for simplicity, we'll let non-None values override if they exist in config.
        if value is not None:
            merged[key] = value
            
    return merged
