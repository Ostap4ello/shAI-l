from pathlib import Path
import json

import logging

logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATH_WITHIN_DB = ".index"
CONFIG_SCHEMA = {"model": ""}


def get_empty_config() -> dict:
    return CONFIG_SCHEMA.copy()


def is_config_valid(config: dict) -> bool:
    for k in CONFIG_SCHEMA.keys():
        if not k in config.keys():
            logger.error(f"Could not find field '{k}' in db config")
            return False
    return True


def resolve_index_paths(
    db_path: str, index_path_within_db: str
) -> tuple[Path, Path, Path]:
    index_dir = Path(db_path).expanduser().resolve() / index_path_within_db
    index_path = index_dir / "index"
    meta_path = index_dir / "index.meta.json"
    config_path = index_dir / "config.json"
    return index_path, meta_path, config_path


def get_default_index_path_within_db() -> str:
    return DEFAULT_INDEX_PATH_WITHIN_DB


def save_index_config(config_path: Path, config: dict) -> None:
    if not is_config_valid(config):
        raise RuntimeError(f"Erroroneous db config ({config_path})")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_index_config(config_path: Path) -> dict:
    logger.debug(f"Loading index config from: {config_path}")
    config = {}

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not is_config_valid(config):
        raise RuntimeError(f"Erroroneous db config ({config_path})")

    return config

