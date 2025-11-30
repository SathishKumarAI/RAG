"""
Configuration management for the RAG pipeline.

WHAT: Centralized configuration loading from environment variables, .env files, and YAML/JSON.
WHY: Provides a single source of truth for all configuration, making it easy to manage
     different environments (dev, staging, prod) and override settings.
HOW: Loads config in priority order: environment variables > .env file > YAML/JSON file > defaults.

Usage:
    from utils.config import get_config
    
    config = get_config()
    model_name = config.get("llm.model_name", "gpt-4")
    api_key = config.get("llm.api_key")  # Raises if not set
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv


class Config:
    """
    Configuration manager that loads settings from multiple sources.
    
    Priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. YAML/JSON config file
    4. Default values (lowest priority)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML/JSON config file (optional)
        """
        self._config: Dict[str, Any] = {}
        
        # Load .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
        
        # Load YAML/JSON config file
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                self._load_config_file(config_path)
        
        # Load environment variables (they override file config)
        self._load_env_vars()
    
    def _load_config_file(self, config_path: Path) -> None:
        """Load configuration from YAML or JSON file."""
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                self._config = yaml.safe_load(f) or {}
            elif config_path.suffix == ".json":
                self._config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # Convert flat env vars to nested dict structure
        # e.g., LLM_API_KEY -> config["llm"]["api_key"]
        for key, value in os.environ.items():
            if key.startswith("RAG_"):
                # Remove RAG_ prefix and convert to nested structure
                parts = key[4:].lower().split("_")
                self._set_nested(parts, value)
    
    def _set_nested(self, parts: list, value: Any) -> None:
        """Set a nested dictionary value from a list of keys."""
        d = self._config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., "llm.model_name")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get("llm.model_name", "gpt-4")
            config.get("vector_store.index_name")
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def get_required(self, key: str) -> Any:
        """
        Get a required configuration value (raises if not found).
        
        Args:
            key: Dot-separated key
            
        Returns:
            Configuration value
            
        Raises:
            ValueError: If key not found
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required configuration key not found: {key}")
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Dot-separated key
            value: Value to set
        """
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self._config.copy()


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get the global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        # Try default config path if not provided
        if config_path is None:
            default_paths = [
                Path("rag-pipeline/configs/config.yaml"),
                Path("configs/config.yaml"),
                Path("config.yaml"),
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
        
        _config_instance = Config(config_path)
    return _config_instance


def reset_config() -> None:
    """Reset the global config instance (useful for testing)."""
    global _config_instance
    _config_instance = None

