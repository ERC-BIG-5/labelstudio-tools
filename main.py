import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.command.annotations import annotations_app, agreements
from ls_helper.command.backup import backup_app
from ls_helper.command.labeling_conf import labeling_conf_app
from ls_helper.command.pipeline import pipeline_app
from ls_helper.command.project_setup import project_app
from ls_helper.command.setup import (
    setup_app,
)
from ls_helper.command.task import task_add_predictions, task_app
from ls_helper.command.view import view_app
from ls_helper.models.interface_models import IChoices
from ls_helper.models.main_models import (
    get_project,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.tasks import strict_update_project_task_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(
    name="Labelstudio helper", pretty_exceptions_show_locals=False
)

app.add_typer(setup_app, name="setup", short_help="Commands related to initializing the project")
app.add_typer(project_app, name="project", short_help="Commands related to project setup and maintenance")
app.add_typer(backup_app, name="backup", short_help="Commands related to bulkbacking up projects and annotations")
app.add_typer(labeling_conf_app, name="labeling-conf",
              short_help="Commands related to building, validating and uploading project label configurations")
app.add_typer(task_app, name="task", short_help="Commands related to downloading, creating and patching project tasks")
app.add_typer(view_app, name="view", short_help="Commands related to project views")
app.add_typer(annotations_app, name="annotations",
              short_help="Commands related to downloading and analyzing annotations")
app.add_typer(pipeline_app, name="pipeline", short_help="Commands related to interaction with the Pipeline package")



@app.command(
    short_help="[ls maint] Update tasks. Files must be matching lists of {id: , data:}"
)
# todo: more testing
def strict_update_project_tasks(
        new_data_file: Path,
        existing_data_file: Optional[Path] = None
):
    raise NotImplementedError("client.patch_task parameters changed")
    new_data_list = json.loads(new_data_file.read_text(encoding="utf-8"))
    if existing_data_file:
        existing_data_list = json.loads(
            existing_data_file.read_text(encoding="utf-8")
        )
        assert len(new_data_list) == len(existing_data_list)

        for idx, t in tqdm(enumerate(new_data_list)):
            t_id = t["id"]
            ex_t = existing_data_list[idx]
            assert t_id == ex_t["id"]
            strict_update_project_task_data(t_id, t, ex_t)

        print(f"{len(new_data_list)} tasks updated")
        return

    for t in tqdm(new_data_list):
        ls_client().patch_task(t["id"], t["data"])

    print(f"{len(new_data_list)} tasks updated")


def get_all_variable_names(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    # todo redo and test...
    struct = po.raw_interface_struct
    return list(struct.ordered_fields_map.keys())


def get_variables_info(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    return [
        {
            "name": k,
            "required": v.required,
            "choice_type": v.choice if isinstance(v, IChoices) else None,
        }
        for k, v in po.raw_interface_struct.ordered_fields_map.items()
    ]


def add_prediction_test():
    resp = task_add_predictions(
        33030,
        {
            "model_version": "one",
            "score": 0.5,
            # "type": "choices",
            "result": [
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_any",
                    "value": {"choices": ["Yes"]},
                },
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_visual",
                    "value": {"choices": ["Yes"]},
                },
            ],
        },
    )
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    twitter = "twitter"
    youtube = "youtube"
    en = "en"
    es = "es"
    tw_es = {"platform": twitter, "language": es}
    yt_en4 = {"id": 50}
    _default = tw_es

    # setup
    from ls_helper.command import setup

    setup.add_projects()

    # this will work, since there is just one spanish twitter (so its set to default)
    agreements(**{"alias": "twitter-en-3"}, variables=["nature_any"])
