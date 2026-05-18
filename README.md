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
    # Create config in $HOME/.config/shai
    shai --create-config
    # OR create config in specified file
    shai --create-config --create <path-to-file>
    # Create default db dir (if not overriden in config/flags)
    mkdir -p ~/.local/share/shai_db
    # Pull models used by shai (see model/embed_model fields in config)
    shai shart_ollama --create
    shai utils pull_model <model/embed_model>
    ```


<br>

## Usage

The application provides a command-line interface (CLI) for interacting with various functionalities.

```bash
shai -h
```

This functionality can be accessed via these CLI subcommands, which are described below:
- `llm` - Direct LLM interactions (generation, embedding)
- `db` - Database indexing and retrieval
- `rag` - RAG-enabled generation
- `utils` - Miscellaneous utilities (Ollama management, man page conversion)

<br>

## Development

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
    - Performance tests: read [test-ctf readme](https://github.com/Ostap4ello/shAI-l/blob/master/test-ctf/README.md)  
    <br>

- **Run Modules Directly**:
If you're developing iv .venv, you can run modules directly:

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
    python -m shAI_ostap4ello.src.utils start_ollama
    python -m shAI_ostap4ello.src.utils stop_ollama
    python -m shAI_ostap4ello.src.utils is_ollama_running
    python -m shAI_ostap4ello.src.utils convert_man_pages --src-dir \<src\> --out-dir \<out\>
    python -m shAI_ostap4ello.src.utils fetch_man_db
    ```

## License

GPL-3.0-or-later
