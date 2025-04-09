import json
from pathlib import Path
from typing import Callable, Annotated

import typer

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import TaskCreate
from ls_helper.new_models import platforms_overview
from tools.project_logging import get_logger

logger = get_logger(__file__)

pipeline_app = typer.Typer(name="Pipeline config", pretty_exceptions_show_locals=True)
_app = pipeline_app

# todo duplicate in main
_project: Callable = lambda id, al, pl, la: platforms_overview.get_project((id, al, pl, la))

_app.command()


def create_task(
        src_path: Annotated[Path, typer.Argument()],
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
):
    po = _project(id, alias, platform, language)
    t = TaskCreate(project=po.id, data=json.loads(src_path.read_text())["data"])
    ls_client().create_task(t)


def create_tasks(
        src_path: Annotated[Path, typer.Argument()],
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
):
    """
    can deal with json files (list of tasks) or a folder, where each task is in its own file.
    It will always turn it into a batch and use the projects-import endpoint
    in case of a folder,
    :param src_path:
    :param batch_import:
    :param id:
    :param alias:
    :param platform:
    :param language:
    :return:
    """
    po = _project(id, alias, platform, language)
    batch: list[TaskCreate] = []
    if src_path.exists():
        # load individual tasks
        if src_path.is_dir():
            for t_f in src_path.glob("*.json"):
                batch.append(TaskCreate(project=po.id, data=json.loads(t_f.read_text())["data"]))
        else:
            batch = [TaskCreate(project=po.id, data=t["data"]) for t in json.loads(src_path.read_text())]

        ls_client().import_tasks(po.id, batch)
