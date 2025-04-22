# Readme

Required repository:
`https://github.com/transfluxus/python-project-tools`

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