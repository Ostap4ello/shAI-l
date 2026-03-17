import os
import subprocess
import logging

logger = logging.getLogger(__name__)

OLLAMA_DOCKER_SCRIPT = "ollama-docker.sh"
OLLAMA_CONTAINER_DEFAULT_NAME = "ollama-node-1"
OLLAMA_DEFAULT_CONTEXT_LENGTH = 32000
OLLAMA_DEFAULT_GPUS = "all"

COMPILE_GROFF_SCRIPT = "compile-groff.sh"


def _call_bash_script(script, args) -> tuple[int, str, str]:
    script_path = os.path.join(os.path.dirname(__file__), script)
    logger.debug(f"Calling bash script: {script_path} with args: {args}")
    try:
        result = subprocess.run(
            [script_path, *args],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.debug(f"Script {script_path} executed successfully.")
        logger.debug(f"stdout: {result.stdout}, stderr: {result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError as e:
        logger.error(f"Script not found: {script_path}")
        raise RuntimeError(f"Script not found: {script_path}") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing script: {script_path}")
        logger.debug(f"retcode: {e.returncode}, stdout: {e.stdout}, stderr: {e.stderr}")
        raise RuntimeError(
            f"Error executing script: {script_path}. Return code: {e.returncode}, stdout: {e.stdout}, stderr: {e.stderr}"
        ) from e


def run_ollama(
    context_length=OLLAMA_DEFAULT_CONTEXT_LENGTH,
    gpus=OLLAMA_DEFAULT_GPUS,
    name=OLLAMA_CONTAINER_DEFAULT_NAME,
):
    logger.info(f"Running Ollama Docker container")
    logger.debug(f"name: {name}, context_length: {context_length}, gpus: {gpus}")
    result = _call_bash_script(
        OLLAMA_DOCKER_SCRIPT,
        [
            "--name",
            name,
            "run",
            "--context-length",
            str(context_length),
            "--gpus",
            gpus,
        ],
    )
    logger.info(f"Ollama Docker container started successfully.")
    return result


def start_ollama(
    context_length=OLLAMA_DEFAULT_CONTEXT_LENGTH,
    gpus=OLLAMA_DEFAULT_GPUS,
    name=OLLAMA_CONTAINER_DEFAULT_NAME,
):
    logger.info(f"Starting Ollama Docker container")
    logger.debug(f"name: {name}, context_length: {context_length}, gpus: {gpus}")
    result = None
    try:
        result = _call_bash_script("ollama-docker.sh", ["--name", name, "begin"])
        logger.info(f"Ollama Docker container started successfully.")
    except RuntimeError as e:
        logger.warning(f"Failed to start Ollama container. Trying to run it instead.")
        result = run_ollama(context_length=context_length, gpus=gpus, name=name)
    return result


def stop_ollama(name=OLLAMA_CONTAINER_DEFAULT_NAME):
    logger.info(f"Stopping Ollama Docker")
    logger.debug(f"name: {name}")
    return _call_bash_script("ollama-docker.sh", ["--name", name, "stop"])


def is_ollama_running(name=OLLAMA_CONTAINER_DEFAULT_NAME) -> bool:
    logger.debug(f"Checking if Ollama Docker container with name: {name} is running")
    return_code, stdout, stderr = _call_bash_script(
        "ollama-docker.sh", ["--name", name, "status"]
    )
    if stdout == "- Container is up.\n- Ollama is up.\n":
        logger.info("Ollama container is up.")
        return True
    else:
        logger.info("Ollama container is down")
        return False


def convert_man_pages_to_text(src_dir, out_dir):
    logger.info(f"Converting Groff files from {src_dir} to {out_dir}")
    return _call_bash_script("compile-groff.sh", ["-i", src_dir, "-o", out_dir])
