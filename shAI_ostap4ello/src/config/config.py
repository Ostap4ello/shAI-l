#!/usr/bin/env python3

import configparser
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.expanduser("~/.config/shai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.conf")


def get_default_config() -> configparser.ConfigParser:
    """Returns default configuration structure."""
    config = configparser.ConfigParser()

    config["general"] = {
        "keep_ollama_running": "false",
    }

    config["llm"] = {
        "model": "qwen3:1.7b",
        "embed_model": "ibm/granite-embedding:125m",
        "api_base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
    }

    config["db"] = {
        "db_path": "~/.local/share/shai_db",
        "index_path_within_db": ".index",
        "batch_size": "32",
        "top_k": "5",
    }

    config["rag"] = {
        "top_k": "5",
        "model": "qwen3:1.7b",
    }

    config["utils"] = {
        "ollama_context_length": "32000",
        "ollama_gpus": "all",
        "ollama_container_name": "ollama-node-1",
    }

    return config


def ensure_config_exists() -> None:
    """Create config file with defaults if it doesn't exist."""
    if os.path.exists(CONFIG_FILE):
        logger.debug(f"Config file already exists: {CONFIG_FILE}")
        return

    logger.info(f"Creating default config file: {CONFIG_FILE}")
    os.makedirs(CONFIG_DIR, exist_ok=True)

    config = get_default_config()
    with open(CONFIG_FILE, "w") as f:
        config.write(f)

    logger.info(f"Config file created: {CONFIG_FILE}")


def load_config() -> configparser.ConfigParser:
    """Load configuration from file, creating it with defaults if it doesn't exist."""
    ensure_config_exists()

    config = configparser.ConfigParser()
    try:
        config.read(CONFIG_FILE)
        logger.debug(f"Loaded config from: {CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Error reading config file: {e}. Using defaults.")
        config = get_default_config()

    return config


def get_config_value(section: str, key: str, fallback: Any = None) -> str:
    """Get a single config value with fallback."""
    config = load_config()
    try:
        return config.get(section, key, fallback=fallback)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback
