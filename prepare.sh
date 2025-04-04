#!/bin/bash

. ../platforms-clients/.venv/bin/activate

MAIN_DIR="/home/rsoleyma/projects"
databases_REPO_NAME=Databases
tools_REPO_NAME=Tools

export PYTHONPATH="${PYTHONPATH}:."
export PYTHONPATH="${PYTHONPATH}:${MAIN_DIR}/$databases_REPO_NAME"
export PYTHONPATH="${PYTHONPATH}:${MAIN_DIR}/$tools_REPO_NAME"
#typer src/main.py run

echo "run: typer src/main.py run ..."
echo "or: typer src/main.py run --help"


#export PYTHONPATH=.