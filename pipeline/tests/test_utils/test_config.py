"""Tests for config utility."""

import pytest
from pathlib import Path
import tempfile
import os

from src.utils.config import Config, get_config, reset_config


def test_config_basic():
    """Test basic config functionality."""
    config = Config()
    config.set("test.key", "value")
    assert config.get("test.key") == "value"
    assert config.get("test.nonexistent", "default") == "default"


def test_config_env_vars(monkeypatch):
    """Test config loading from environment variables."""
    monkeypatch.setenv("RAG_LLM_MODEL", "gpt-4")
    config = Config()
    assert config.get("llm.model") == "gpt-4"


def test_config_required():
    """Test required config key."""
    config = Config()
    config.set("required.key", "value")
    assert config.get_required("required.key") == "value"
    
    with pytest.raises(ValueError):
        config.get_required("nonexistent.key")


def test_config_yaml(tmp_path):
    """Test config loading from YAML file."""
    import yaml
    
    config_file = tmp_path / "config.yaml"
    config_data = {"llm": {"model": "gpt-4", "temperature": 0.7}}
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    config = Config(config_path=config_file)
    assert config.get("llm.model") == "gpt-4"
    assert config.get("llm.temperature") == 0.7


def test_get_config_singleton():
    """Test get_config singleton pattern."""
    reset_config()
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2

