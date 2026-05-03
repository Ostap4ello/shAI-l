import os
import subprocess
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

OLLAMA_DOCKER_SCRIPT = "ollama-docker.sh"
OLLAMA_CONTAINER_DEFAULT_NAME = "ollama-node-1"
OLLAMA_DEFAULT_CONTEXT_LENGTH = 32000
OLLAMA_DEFAULT_GPUS = "all"

COMPILE_GROFF_SCRIPT = "compile-groff.sh"


def _call_bash_script(
    script: str,
    args: list[str],
    stdout_callback: Optional[Callable[[str], None]] = None,
    stderr_callback: Optional[Callable[[str], None]] = None,
) -> tuple[int, str, str]:
    script_path = os.path.join(os.path.dirname(__file__), script)
    logger.debug(f"Calling bash script: {script_path} with args: {args}")

    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        raise RuntimeError(f"Script not found: {script_path}")

    try:
        # Use Popen for real-time streaming
        process = subprocess.Popen(
            [script_path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        stdout_buffer = []
        stderr_buffer = []

        # Read stdout and stderr simultaneously
        import select

        while True:
            # Use select to monitor both streams for readability
            if process.stdout and process.stderr:
                ready, _, _ = select.select(
                    [process.stdout, process.stderr], [], []
                )

                for stream in ready:
                    line = stream.readline()
                    if not line:
                        continue

                    if stream == process.stdout:
                        stdout_buffer.append(line)
                        logger.debug(f"stdout: {line.rstrip()}")
                        if stdout_callback:
                            stdout_callback(line.rstrip())
                    elif stream == process.stderr:
                        stderr_buffer.append(line)
                        logger.debug(f"stderr: {line.rstrip()}")
                        if stderr_callback:
                            stderr_callback(line.rstrip())

            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                if process.stdout:
                    remaining_stdout = process.stdout.readlines()
                    stdout_buffer.extend(remaining_stdout)
                    for line in remaining_stdout:
                        logger.debug(f"stdout: {line.rstrip()}")
                        if stdout_callback:
                            stdout_callback(line.rstrip())

                if process.stderr:
                    remaining_stderr = process.stderr.readlines()
                    stderr_buffer.extend(remaining_stderr)
                    for line in remaining_stderr:
                        logger.debug(f"stderr: {line.rstrip()}")
                        if stderr_callback:
                            stderr_callback(line.rstrip())
                break

        return_code = process.returncode
        stdout_str = "".join(stdout_buffer)
        stderr_str = "".join(stderr_buffer)

        if return_code != 0:
            logger.error(f"Error executing script: {script_path}")
            logger.debug(
                f"retcode: {return_code}, stdout: {stdout_str}, stderr: {stderr_str}"
            )
            raise RuntimeError(
                f"Error executing script: {script_path}. Return code: {return_code}, stdout: {stdout_str}, stderr: {stderr_str}"
            )

        logger.debug(f"Script {script_path} executed successfully.")
        return return_code, stdout_str, stderr_str

    except FileNotFoundError as e:
        logger.error(f"Script not found: {script_path}")
        raise RuntimeError(f"Script not found: {script_path}") from e
    except Exception as e:
        logger.error(f"Error executing script: {script_path}: {e}")
        raise RuntimeError(f"Error executing script: {script_path}: {e}") from e


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
    def stdout_callback(s):
        logger.info(s)
    return _call_bash_script("compile-groff.sh", ["-i", src_dir, "-o", out_dir], stdout_callback=stdout_callback)
