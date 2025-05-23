import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tools.project_logging import get_logger
from tqdm import tqdm

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    Task as LSTask,
)
from ls_helper.my_labelstudio_client.models import (
    TaskCreate as LSTaskCreate,
)
from ls_helper.my_labelstudio_client.models import (
    TaskCreateList as LSTaskCreateList,
)
from ls_helper.my_labelstudio_client.models import (
    TaskList as LSTaskList,
)
from ls_helper.models.main_models import ProjectData, get_project

logger = get_logger(__file__)

task_app = typer.Typer(name="Task related", pretty_exceptions_show_locals=True)


def task_platform_id_map(
    tasks: LSTaskList | LSTaskCreateList,
) -> dict[str, LSTask]:
    platform_id_map: dict[str, LSTask] = {}
    for task in tasks.root:
        p_id = task.data.get("platform_id")
        if not p_id:
            raise ValueError("Task data does not have platform_id!!!")
        platform_id_map[p_id] = task

    return platform_id_map


@task_app.command()
def create_task(
    src_path: Annotated[Path, typer.Argument()],
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
):
    po = get_project(id, alias, platform, language)
    t = LSTaskCreate(
        project=po.id, data=json.loads(src_path.read_text())["data"]
    )
    ls_client().create_task(t)


@task_app.command()
def create_tasks(
    src_path: Annotated[Path, typer.Argument()],
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
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
    po = get_project(id, alias, platform, language)
    batch: list[LSTaskCreate] = []
    if not src_path.exists():
        print(f"Path not found: {src_path}")

    # load individual tasks
    if src_path.is_dir():
        for t_f in src_path.glob("*.json"):
            batch.append(
                LSTaskCreate(
                    project=po.id, data=json.loads(t_f.read_text())["data"]
                )
            )
    else:
        batch = [
            LSTaskCreate(project=po.id, data=t["data"])
            for t in json.loads(src_path.read_text())
        ]

    resp_data = ls_client().import_tasks(po.id, batch)
    task_ids = resp_data["task_ids"]
    tasks = [
        LSTask(**t.model_dump(), id=task_ids[idx])
        for idx, t in enumerate(batch)
    ]
    po.save_tasks(tasks)


def patch_task(task_id: int, task: LSTask):
    """very similar to creating. But actually uses the patch api endpoint"""
    # todo, we can try if the data validates as LSTask. Meaning they have an id already.
    # we can then skip the mapping part...
    res = ls_client().patch_task(task_id, task)
    return res


def task_add_predictions(task_id: int, data):
    return ls_client().add_prediction(task_id, data)


@task_app.command()
def patch_tasks(
    src_path: Annotated[Path, typer.Argument()],
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
):
    """very similar to creating. But actually uses the patch api endpoint"""
    # todo, we can try if the data validates as LSTask. Meaning they have an id already.
    # we can then skip the mapping part...
    po = get_project(id, alias, platform, language)
    batch: LSTaskCreateList | LSTaskList = LSTaskCreateList(root=[])
    if src_path.exists():
        # load individual tasks
        if src_path.is_dir():
            for t_f in src_path.glob("*.json"):
                batch.root.append(
                    LSTaskCreate(
                        project=po.id, data=json.loads(t_f.read_text())["data"]
                    )
                )
        else:
            batch = LSTaskCreateList(
                root=[
                    LSTaskCreate(project=po.id, data=t["data"])
                    for t in json.loads(src_path.read_text())
                ]
            )

    update_map = task_platform_id_map(batch)

    tasks = po.get_tasks()
    platform_id_map = task_platform_id_map(tasks)

    # TODO,  can we do this async in order to speed things up?
    for p_id, update_task in tqdm(update_map.items()):
        ls_task = platform_id_map.get(p_id)
        if not ls_task:
            print(
                f"Warning: platform_id: {p_id} not present in project tasks: {repr(tasks)}"
            )
        task_id = ls_task.id
        # update_task.id = task_id
        _ = ls_client().patch_task(task_id, update_task)
        # TODO store them


@task_app.command()
def get_tasks(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
) -> list[LSTask]:
    po: ProjectData = get_project(id, alias, platform, language)
    tasks = ls_client().get_task_list(project=po.id)
    po.save_tasks(tasks)
    return tasks


@task_app.command()
def get_task(
    id: Annotated[Optional[int], typer.Option()] = None,
) -> LSTask:
    resp = ls_client().get_task(id)
    if resp.status_code != 200:
        print(resp.status_code, resp.json())
    return LSTask.model_validate(resp.json())
