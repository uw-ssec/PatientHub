# Patient Simulation Project

> A framework for simulating patients

## Directory

- `frontend/` includes the code for the user interface (using Streamlit)
- `backend/` includes the code for agents (using LangChain) + API server (FastAPI)

## Requirements

- Install `uv` -> `pip install uv`
- Create virtual environment -> `uv venv patient-simulation --python 3.13`
- Install dependencies -> `uv sync`
- Load virtual environment -> `source .venv/bin/activate`

## Usage

> For now, we're mainly working on the simulation part on the backend (backend/src)

```bash
cd backend
uv run src/simulate.py
```



