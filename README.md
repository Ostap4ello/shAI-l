# ShAI - minimal shell assistant powered by local llm.

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### CLI Commands
The application provides a command-line interface (CLI) for interacting with the LLM. Below are the available commands:

#### `generate`
Generate text responses from an LLM.
```bash
python -m src.llm --model <llm_model_name> "<prompt text>" --stream
```
- `--model`: Name of the model (e.g., "gpt-3.5-turbo"). Default: `qwen3:1.7b`.
- `--stream`: (optional) Enable streaming output.

#### `build`
Create an index from documents.
```bash
python -m src.llm build --db-path ./documents --index-path-within-db .index
```

#### `search`
Search previously saved indexes for relevant data:
```bash
python -m src.llm search --top-k <int> "search query string"
```
