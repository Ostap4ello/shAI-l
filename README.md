# ShAI - Minimal Shell Assistant Powered by Local LLM

This is a minimalistic shell assistant that leverages athe technologies of local Large Language Models (LLMs) to provide various functionalities, such as generating text responses, creating embeddings, and performing Retrieval-Augmented Generation (RAG). The assistant is designed to be lightweight and easy to use, with a focus on local execution without relying on external APIs.

**Table of contents:**
1. [Setup](#setup)
1. [Usage](#usage)
1. [Development](#development)

<br>

## Setup

1. Install the following system dependencies:
    - `python3`
    - `pipx`
    - `groff`
    - `jq`
    - `docker` (for running Ollama) ([instructions](https://docs.docker.com/desktop/setup/install/linux/)).

2. Run setup commands:
    ```bash
    # Allow current user to execute docker without `sudo`
    sudo usermod -aG docker $USER
    # Pull ollama docker image
    docker pull ollama/ollama:latest
    ```

3. Install shAI:
    - As a package (Recommended)
        ```bash
        git clone https://github.com/Ostap4ello/shAI-l
        cd shAI-l

        pipx install .
        ```
        This will install the `shai` CLI command globally.
    - In .venv (Recommended)
        ```bash
        git clone https://github.com/Ostap4ello/shAI-l
        cd shAI-l

        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e .
        ```
        This will install the `shai` CLI command within curent .venv.

<br>

4. Run shAI-related setup:
    ```
    # Create config in $HOME/.config/shai (or use --config flag to specify the path)
    shai --create-config
    # Pull models used by shai (see model/embed_model fields in config)
    shai init
    ```


<br>

## Usage

The application provides a command-line interface (CLI) for interacting with various functionalities.

```bash
shai -h
```


**NOTE 1:** Ensure that the ollama docker image is running before using shAI, as it relies
on it for LLM interactions. 
```bash
# Check if ollama is running
shai utils is-ollama-running
# Start ollama if it's not running
shai utils start-ollama
```

The application can be configured via configuration file located at
`$HOME/.config/shai/config.yaml` (or specified via `--config` flag). The
configuration file allows you to set parameters for LLM interactions, database
settings, and other options. You can also use environment variables to override
specific configuration values.

**NOTE 2:** After updating the models in the configuration file, run `shai init`
to pull the new models, or do it manually via `shai utils pull-model <model_name>`.

This functionality can be accessed via these CLI subcommands, which are described below:
- `workflows` - Predefined workflows that combine multiple functionalities for
  specific use cases
- `llm` - Direct LLM interactions (generation, embedding)
- `db` - Database indexing and retrieval
- `rag` - RAG-enabled generation
- `utils` - Miscellaneous utilities (Ollama/Docker management, man page conversion)

<br>

## Development & Contribution

- **Application architecture** is described in [docs/architecture.md](https://github.com/Ostap4ello/shAI-l/blob/master/docs/architecture.md)  

- **Development installation**: Install the application in .venv (see [setup](#setup)).

- **Building the .tar.gz package**:
    ```bash
    python -m build
    ```
    This will create a `dist/` directory with the built package, which can be installed using pip/pipx:
    ```bash
    pipx install dist/<package_name>.tar.gz
    ```

- **Testing**:
    - Performance or Regression tests: read [test readme](https://github.com/Ostap4ello/shAI-l/blob/master/test/README.md)  
    <br>

- **Run Modules Directly**:
If you're developing in .venv, you can run modules directly:

    ```bash
    # Main CLI
    python -m shAI_ostap4ello.src -h

    # LLM module
    python -m shAI_ostap4ello.src.llm generate "Hello world"
    python -m shAI_ostap4ello.src.llm embed "Some text"

    # Database module
    python -m shAI_ostap4ello.src.db build --db-path ~/.local/share/shai_db
    python -m shAI_ostap4ello.src.db search "search query"
    python -m shAI_ostap4ello.src.db check

    # RAG module
    python -m shAI_ostap4ello.src.rag find "query"

    # Utils module
    python -m shAI_ostap4ello.src.utils start-ollama
    python -m shAI_ostap4ello.src.utils stop-ollama
    python -m shAI_ostap4ello.src.utils is-ollama_running
    python -m shAI_ostap4ello.src.utils convert-man-pages --src-dir \<src\> --out-dir \<out\>
    python -m shAI_ostap4ello.src.utils fetch-man-db
    ```

## License

GPL-3.0-or-later
