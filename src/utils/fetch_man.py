from typing import Set
import logging
import os
import shutil
import tempfile

from .adapter import convert_man_pages_to_text, _call_bash_script

logger = logging.getLogger(__name__)

MAN_ROOT = "/usr/share/man"
ALLOWED_SECTIONS = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
DEFAULT_SECTIONS = {"1", "3", "5", "8"}
MERGE_POLICIES = {"abort", "clean", "merge-ours", "merge-theirs"}


def fetch_manpages_to_db(
    db_path: str,
    sections: Set[str] = DEFAULT_SECTIONS,
    man_dir: str = MAN_ROOT,
    merge_strategy: str = "abort",
):
    db_root = db_path

    if os.path.exists(db_root):
        if merge_strategy == "abort":
            not_empty = bool(os.listdir(db_root))
            if not_empty:
                raise RuntimeError(
                    f"DB dir '{db_path}' is not empty (merge_strategy=abort)"
                )
        elif merge_strategy == "clean":
            shutil.rmtree(db_root, ignore_errors=True)
            # Other merge strategies handled below

    os.makedirs(db_root, exist_ok=True)

    if not os.path.exists(man_dir):
        raise RuntimeError(f"Man directory '{man_dir}' does not exist")

    tmp_dir_source = tempfile.mkdtemp(prefix="man_s_")
    tmp_dir_compiled = tempfile.mkdtemp(prefix="man_c_")
    logger.debug(f"Created temporary dirs: source={tmp_dir_source}, compiled={tmp_dir_compiled}")

    for section in sections:
        sys_man_dir = os.path.join(MAN_ROOT, f"man{section}")
        if not os.path.exists(sys_man_dir):
            logger.warning(f"Warning: section {section} not found at {sys_man_dir}")
            continue
        # Recursively copy sys_man_dir to tmp_dir_source
        shutil.copytree(sys_man_dir, os.path.join(tmp_dir_source, f"man{section}"), dirs_exist_ok=True, symlinks=True)

        try:
            convert_man_pages_to_text(src_dir=tmp_dir_source, out_dir=tmp_dir_compiled)
        except Exception as e:
            logger.warning(f"Error converting man pages: {e}")
            shutil.rmtree(tmp_dir_source, ignore_errors=True)
            shutil.rmtree(tmp_dir_compiled, ignore_errors=True)
            raise RuntimeError(f"Error converting man pages")

    for root, _, files in os.walk(tmp_dir_compiled):
        for file in files:
            if not file.endswith('.txt'):
                continue
            txt_file = os.path.join(root, file)
            relative_path = os.path.relpath(txt_file, tmp_dir_compiled)
            dest_file = os.path.join(db_root, relative_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            if os.path.exists(dest_file):
                if merge_strategy == "merge-ours":
                    logger.debug(f"Skipping existing file (merge-ours): {dest_file}")
                    continue
                elif merge_strategy == "merge-theirs":
                    logger.debug(f"Overwriting existing file (merge-theirs): {dest_file}")
                    pass
                else:
                    logger.error(
                        f"File already exists unexpectedly: {dest_file} (merge_strategy={merge_strategy})"
                    )
                    shutil.rmtree(tmp_dir_source, ignore_errors=True)
                    shutil.rmtree(tmp_dir_compiled, ignore_errors=True)
                    shutil.rmtree(db_root, ignore_errors=True)
                    raise RuntimeError(f"Error merging man pages")
            with open(txt_file, "rb") as src, open(dest_file, "wb") as dst:
                dst.write(src.read())

    shutil.rmtree(tmp_dir_source, ignore_errors=True)
    shutil.rmtree(tmp_dir_compiled, ignore_errors=True)
    logger.info(f"Man pages fetched and converted to text database at: {db_path}")
