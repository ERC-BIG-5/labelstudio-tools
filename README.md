# Readme

## Installation

Make sure you have `python-dev`installed for your version installed on your system. 
e.g. `sudo apt-get install python3.13-dev`.

Install all dependencies with `uv sync` or `uv sync --python python3.XX` (your prefered python version).

Alternatively, create a virtual environment and install the dependencies with pip:
```shell
# in linux
python -m venv .venv
. ./venv/bin/activate
pip install -e .
```

You also need this tools repository:
`https://github.com/transfluxus/python-project-tools`

Currently, that repo, has no pyproject.toml nor setup.py, so it must be cloned and added to the
pythonpath when running scripts from this project.

linux:
```
# e.g. 
PYTHONPATH=../python-project-tools python3 main.py
```

windows:
```
set PYTHONPATH=../python-project-tools
python main.py
```
## Setup

copy the `.template.env` file and name the copy .env
You need to set the hostname of your labelstudio instance and your api key in that file
You can find your API-KEY in your profile 
e.g.

```shell
LS_HOSTNAME=https://MY-FANTASTIC-LS-STUDIO-INSTANCE.org
LS_API_KEY=API-TOKEN
```

## Getting the LS projects ready

Run 
`PYTHONPATH=../python-project-tools:. typer main.py run setup add-projects`

> [!NOTE] 
> you might add the `:.` to the PYTHONPATH because python does not find the main source package.

Next, you might want to download the ls project details of each project (if you don't want to do that individually).

`PYTHONPATH=../python-project-tools:. typer main.py run setup download-all-projects`

## main.py

main uses [Typer](https://typer.tiangolo.com/), so we can run commands from the commandline
projects of labelstudio are identified through the first 4 options: id, alias, platform, language.

Crucial project overview is store in `data/ls_data/projects.json` 

## Commands

Run commands like so:

`typer main.py run status -a twitter-es-4`

Commands sub folders:

`typer main.py run task get-tasks -a twitter-es-4`

- setup: setting up projects
- task: adding and patching LS tasks
- labeling_conf: creating project labeling configs (xml)
- annotations:  downloading annotations
- view: creating, managing views
- pipeline: interacting with the pipeline package
- backup: backing up project data and annotations
    

## ls_data folder
The ls_data folder (`data/ls_data`) contains all relevant downloaded files.
- project: project data as it is retrieved from the LS API
- annotations: annotations downloaded from the ls api.
- agreements:
### Keeping the configs right

People decide on changing the configs while data is coming in.
For this, we have the fixes files (Model: ProjectAnnotationExtension)
which stores "fixes" a dict of names that correspond to task_data (optionally),
choices and text-inputs (other later, other types of user input).

These fixes per variable of type:

```python
class VariableExtension(BaseModel):
    name_fix: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str | list[str]] = None
    split_annotation: Optional[list[VariableSplit]] = None
    deprecated: Optional[bool] = None
```

crucially have: `name_fix` which is a fix applied after creating result and agreement tables.
`default` and `deprecated`

Todo:
- [ ] we need more tooling around `deprecated`. kicking them out of the results and agreements. 
    and finding where they are still in use (in order to create LS views) so people can update the annotations.
    A final step  would be removing them from the label_configs, and fixes, when data is clean.
   the configs, should work with the "hidden" or "deprecated" classes, to hidden, highlight them (red).

## Guidelines

### Creating a new project

`project create-project`
`project setup-project-settings`
this will also create a <id>.xml in the `labeling_configs` folder
`labeling-conf build-ls-labeling-interface`
use the platform-specific template...
`labeling-conf update-labeling-config`


