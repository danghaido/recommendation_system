# KHDL Streamlit App

This is a simple Streamlit application for data handling and visualization.

## Prerequisites

- Python 3.10+ installed
- [uv](https://github.com/astral-sh/uv) installed globally. You just need to run the following command: pip install uv

Create and sync the environment from `pyproject.toml` using uv:

```
Step 1: Create the environment
```uv venv create

Step 2: Sync the environment
```uv sync

Then start Streamlit:
```streamlit run main.py
```