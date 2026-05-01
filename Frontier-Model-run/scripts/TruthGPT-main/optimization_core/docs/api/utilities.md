# Utility Scripts

TruthGPT includes a suite of maintenance and verification scripts to ensure your environment is healthy and your training is observable.

## `health_check.py`

**Usage**: `python utils/health_check.py`

This script performs a comprehensive audit of your environment:
-   **Python Version**: Verifies Python 3.8+.
-   **Dependencies**: Checks for critical libraries (`torch`, `transformers`, `accelerate`).
-   **CUDA**: Verifies GPU availability and CUDA version compatibility.
-   **Core Modules**: Attempts to import internal modules to detect path issues.

## `install_extras.py`

**Usage**: `python install_extras.py [group]`

Manages optional dependencies to keep the core install lightweight.

**Arguments**:
-   `--list`: Show all available dependency groups.
-   `--check`: Show installed status of each group.
-   `[group]`: Install a specific group (e.g., `wandb`, `bitsandbytes`, `test`).
-   `all`: Install everything.

## `monitor_training.py`

**Usage**: `python utils/monitor_training.py runs/my_run`

A lightweight terminal dashboard that watches a running training directory.

**Features**:
-   Detects new checkpoints as they appear.
-   Reads the latest `log` file to show current Loss and Tokens/sec.
-   Monitors system resources (CPU/RAM/GPU).

## `visualize_training.py`

**Usage**: `python utils/visualize_training.py runs/my_run --summary`

Generates post-training reports.

**Arguments**:
-   `--summary`: Prints a high-level summary of the training run (total metrics, best validation loss).
-   `--checkpoints`: Lists all saved checkpoints and their metadata.
-   `--plot`: (Optional) Generates a generic localized plot image of the loss curve.
