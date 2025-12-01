# PatientHub

> A unified hub to create, simulate, and evaluate methods for patient/client simulation.

# Supported Agents

## Clients

- [x] Custom agent (`basic`) -> for testing purposes
- [x] PatientPsi (`patientPsi`) -> from https://arxiv.org/abs/2405.19660
- [x] RoleplayDoh -> from https://aclanthology.org/2024.emnlp-main.591/
- [x] Eeyore -> from https://aclanthology.org/2025.findings-acl.707.pdf
- [ ] EvoPatient -> from https://github.com/ZJUMAI/EvoPatient

# Guide

- Set up and activate the virtual environment (after installling [uv](https://docs.astral.sh/uv/getting-started/installation/))

```bash
uv sync
source .venv/bin/activate
```

- Create a file named `.env` and fill it with the following values

```bash
LAB_API_KEY=<your API key>
LAB_BASE_URL=http://115.182.62.174:18888/v1
```

- Run the following script for simulation

```bash
uv run python -m src.scripts.simulate
```
