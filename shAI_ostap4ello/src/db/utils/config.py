from pathlib import Path
import json

import logging

logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATH_WITHIN_DB = ".index"


def resolve_index_paths(
    db_path: str, index_path_within_db: str
) -> tuple[Path, Path, Path]:
    index_dir = Path(db_path) / index_path_within_db
    index_path = index_dir / "index"
    meta_path = index_dir / "index.meta.json"
    config_path = index_dir / "config.json"
    return index_path, meta_path, config_path


def get_default_index_path_within_db() -> str:
    return DEFAULT_INDEX_PATH_WITHIN_DB


def save_index_config(config_path: Path, model: str) -> None:
    config_path.write_text(json.dumps({"model": model}, indent=2), encoding="utf-8")


def load_index_config(config_path: Path) -> dict:
    logger.info(f"Loading index config from: {config_path}")
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}. Returning empty config.")
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse config: {config_path}. Error: {exc}")
        raise RuntimeError(f"Invalid index config: {config_path}") from exc
