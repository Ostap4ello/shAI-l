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
        "log_level": "INFO",
    }

    config["llm"] = {
        "model": "qwen2.5:1.5b",
        "embed_model": "ibm/granite-embedding:125m",
        "api_base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
    }

    config["db"] = {
        "db_path": "~/.local/share/shai_db",
        "index_path_within_db": ".index",
        "batch_size": "32",
        "show_contents": "false",

        "top_k": "5",
        "section_rows": "0",

        "extended_search": "false",
        "top_k_extended": "10",
        "extended_section_rows": "20",
    }

    config["utils"] = {
        "ollama_url": "http://127.0.0.1:11434/",
        "ollama_context_length": "32000",
        "ollama_gpus": "all",
        "ollama_container_name": "ollama-node-1",
        "merge_strategy": "abort",
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

        logger.info(f"Creating new {config_path} with default values.")
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
    default_log_level = config.get("general", "log_level", fallback="INFO")
    llm_main.DEFAULT_LOG_LEVEL = default_log_level
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
    db_main.DEFAULT_LOG_LEVEL = default_log_level
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
    db_main.DEFAULT_SECTION_ROWS = config.getint(
        "db", "section_rows", fallback=0
    )
    db_main.DEFAULT_EXTS = config.getboolean(
        "db", "extended_search", fallback=False
    )
    db_main.DEFAULT_EXTS_TOP_K = config.getint(
        "db", "top_k_extended", fallback=10
    )
    db_main.DEFAULT_EXTS_SECTION_ROWS = config.getint(
        "db", "extended_section_rows", fallback=20
    )
    db_main.DEFAULT_SHOW_CONTENTS = config.getboolean(
        "db", "show_contents", fallback=False
    )

    # Propagate RAG defaults (from default config)
    from ..workflows import __main__ as ws_main
    ws_main.DEFAULT_LOG_LEVEL = default_log_level

    from ..utils import __main__ as utils_main
    utils_main.DEFAULT_LOG_LEVEL = default_log_level
    utils_main.DEFAULT_DOCKER_CONTAINER_NAME = config.get(
        "utils", "ollama_container_name", fallback="ollama-node-1"
    )
    utils_main.DEFAULT_DOCKER_CONTEXT_LENGTH = config.getint(
        "utils", "ollama_context_length", fallback=32000
    )
    utils_main.DEFAULT_DOCKER_GPUS = config.get(
        "utils", "ollama_gpus", fallback="all"    )
    utils_main.DEFAULT_FETCH_DB_PATH = config.get(
        "db", "db_path", fallback="~/.local/share/shai_db"
    )
    utils_main.DEFAULT_FETCH_MERGE_STRATEGY = config.get(
        "utils", "merge_strategy", fallback="abort"
    )
    utils_main.DEFAULT_OLLAMA_URL = config.get(
        "utils", "ollama_url", fallback="http://127.0.0.1:11434/"
    )

    from .. import __main__ as src_main
    src_main.DEFAULT_LOG_LEVEL = default_log_level

    from ..interpreter import __main__ as interpreter_main
    interpreter_main.DEFAULT_LOG_LEVEL = default_log_level

    logger.debug("Config propagated to module defaults.")
