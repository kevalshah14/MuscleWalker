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
   # For headless simulation (no visual output)
   uv run main.py
   
   # For visual simulation (requires mjpython on macOS)
   uv run mjpython main.py
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

## MuJoCo Visualization

This project includes a bipedal walker simulation with visual output capabilities:

### Visual Simulation
To see the MuJoCo environment in action:
```bash
uv run mjpython main.py
```

This will open a 3D viewer window showing:
- A bipedal walker with torso, thighs, shins, and feet
- Real-time physics simulation
- Random control inputs applied to joints
- Interactive 3D camera controls

### Headless Simulation
For running without visual output:
```bash
uv run main.py
```

### Viewer Controls
When the visual window is open:
- **Rotate**: Click and drag with left mouse button
- **Zoom**: Scroll mouse wheel
- **Pan**: Click and drag with right mouse button
- **Exit**: Press ESC or close the window

## Requirements

- Python 3.13+
- UV package manager
- MuJoCo 3.3.6+ (includes mjpython for macOS visualization)
