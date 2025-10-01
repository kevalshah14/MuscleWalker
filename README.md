## Quick Start

First, install UV if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Project Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone https://github.com/kevalshah14/MuscleWalker.git
   cd MuscleWalker
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the project:**
   ```bash
   uv run main.py
   ```

### Common UV Commands

- **Install dependencies:** `uv sync`
- **Add a new dependency:** `uv add package-name`
- **Add a dev dependency:** `uv add --dev package-name`
- **Remove a dependency:** `uv remove package-name`
- **Run a script:** `uv run script.py`
- **Run with specific Python version:** `uv run --python 3.13 main.py`
- **Create virtual environment:** `uv venv`
- **Activate virtual environment:** `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)

## Requirements

- Python 3.13+
- UV package manager
