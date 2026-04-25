#!/usr/bin/env python3

from pathlib import Path
from typing import Any
import configparser
import logging
from sys import exit

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "~/.config/shai/config.conf"

global config  # Cache for loaded config


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
        "top_k_extended": "10",
        "extended_search": "false",
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


def load_config(
    config_path_str: str = DEFAULT_CONFIG_PATH, create: bool = False
) -> None:
    """Load configuration from file, creating it with defaults if it doesn't exist."""

    global config

    config_path = Path(config_path_str).expanduser()
    if create:
        default_config = get_default_config()
        if config_path.exists():
            logger.error(f"File {config_path} already exists.")
            exit(1)

        logger.info(f" Creating new {config_path} with default values.")
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config.write(config_path.open("w"))
            logger.info(f"Default config created at {config_path}")
        except Exception as e:
            logger.error(f"Failed to create default config file at {config_path}: {e}")
            exit(1)

    tmp_config = get_default_config()
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        exit(1)

    try:
        tmp_config.read(config_path)
        logger.debug(f"Loaded config from: {config_path}")
    except Exception as e:
        logger.warning(f"Error reading config file: {e}. Using defaults.")
        exit(1)

    config = tmp_config
    propagate_config()


def get_config_value(
    section: str, key: str, val_type: type = str, fallback: Any = None
) -> Any:
    """Get a single config value with fallback."""

    global config
    if config is None:
        raise ValueError("Config not loaded.")

    if fallback is not None and not isinstance(fallback, val_type):
        raise ValueError(
            f"Fallback value type {type(fallback)} does not match expected type {val_type}"
        )

    ret = None
    if val_type is int:
        ret = config.getint(section, key, fallback=None)
    elif val_type is bool:
        ret = config.getboolean(section, key, fallback=None)
    elif val_type is float:
        ret = config.getfloat(section, key, fallback=None)
    elif val_type is str:
        ret = config.get(section, key, fallback=None)
    elif val_type is list:
        value = config.get(section, key, fallback=None)
        ret = value.split(",") if value else []
    else:
        raise ValueError(f"Unsupported config value type: {val_type}")

    if ret is None:
        if fallback is None:
            raise ValueError(f"Requested non-existing [{section}] {key}.")

        logger.debug(
            f"Config value [{section}] {key} not found. Using fallback: {fallback}. " +
            "Please use add this to default config file."
        )
        ret = fallback

    return ret


def propagate_config() -> None:
    """Propagate config values to module-level DEFAULT constants.

    Updates DEFAULT_* constants in __main__ modules of llm, db, and rag
    with values from the loaded config. Only propagates values that exist
    in the default config structure.
    """

    global config
    if config is None:
        logger.warning("Config not loaded. Skipping propagation.")
        return

    # Propagate LLM defaults (from default config)
    from ..llm import __main__ as llm_main

    llm_main.DEFAULT_API_BASE_URL = config.get(
        "llm", "api_base_url", fallback="http://127.0.0.1:11434/v1"
    )
    llm_main.DEFAULT_API_KEY = config.get("llm", "api_key", fallback="ollama")
    llm_main.DEFAULT_MODEL = config.get("llm", "model", fallback="qwen3:1.7b")
    llm_main.DEFAULT_EMBED_MODEL = config.get(
        "llm", "embed_model", fallback="ibm/granite-embedding:125m"
    )

    # Propagate DB defaults (from default config)
    from ..db import __main__ as db_main
    db_main.DEFAULT_API_BASE_URL = config.get(
        "llm", "api_base_url", fallback="http://127.0.0.1:11434/v1"
    )
    db_main.DEFAULT_API_KEY = config.get("llm", "api_key", fallback="ollama")
    db_main.DEFAULT_EMBED_MODEL = config.get(
        "llm", "embed_model", fallback="ibm/granite-embedding:125m"
    )
    db_main.DEFAULT_DB_PATH = config.get(
        "db", "db_path", fallback="~/.local/share/shai_db"
    )
    db_main.DEFAULT_INDEX_PATH_WITHIN_DB = config.get(
        "db", "index_path_within_db", fallback=".index"
    )
    db_main.DEFAULT_BATCH_SIZE = config.getint(
        "db", "batch_size", fallback=32
    )
    db_main.DEFAULT_TOP_K = config.getint("db", "top_k", fallback=5)
    db_main.DEFAULT_TOP_K_EXTENDED = config.getint(
        "db", "top_k_extended", fallback=10
    )
    db_main.DEFAULT_EXTENDED_SEARCH = config.getboolean(
        "db", "extended_search", fallback=False
    )

    # Propagate RAG defaults (from default config)
    from ..rag import __main__ as rag_main

    rag_main.DEFAULT_API_BASE_URL = config.get(
        "llm", "api_base_url", fallback="http://127.0.0.1:11434/v1"
    )
    rag_main.DEFAULT_API_KEY = config.get("llm", "api_key", fallback="ollama")
    rag_main.DEFAULT_MODEL = config.get("rag", "model", fallback="qwen3:1.7b")
    rag_main.DEFAULT_EMBED_MODEL = config.get(
        "llm", "embed_model", fallback="ibm/granite-embedding:125m"
    )

    logger.debug("Config propagated to module defaults.")
