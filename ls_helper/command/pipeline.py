import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tools.project_logging import get_logger

from ls_helper.models.main_models import (
    get_project,
)
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS

logger = get_logger(__file__)

pipeline_app = typer.Typer(
    name="Pipeline config", pretty_exceptions_show_locals=True
)
_app = pipeline_app


@_app.command()
def create_pipeline_flow(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    destination: Annotated[Optional[Path], typer.Argument()] = None,
    label_filter: Annotated[Optional[str], typer.Option("-f")] = None,
    add_label: Annotated[Optional[str], typer.Option("-l")] = None,
):
    pass


@_app.command()
def reformat_for_datapipelines(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    destination: Annotated[Optional[Path], typer.Argument()] = None,
    accepted_ann_age: Annotated[
        int, typer.Option(help="Download annotations if older than x hours")
    ] = 6,
):
    """
    create a file with a dict platform_id: annotation, which can be ingested by the pipeline
    :param platform:
    :param language:
    :param destination:
    :return:
    """
    # does extra calculation but ok.
    po = get_project(id, alias, platform, language)
    local, results = ProjectMgmt.get_recent_annotations(
        po.id, accepted_ann_age
    )
    # print(results)

    print(results.stats())
    am = results.drop_cancellations()
    # print(am.stats())
    res = {}
    # print(am.completed())
    for task_result in am.task_results:
        res[task_result.data["platform_id"]] = {
            po.id: task_result.model_dump(exclude={"data"})
        }

    if not destination:
        destination = (
            SETTINGS.temp_file_path
            / f"annotations_for_datapipelines_{po.id}.json"
        )
        destination.write_text(json.dumps(res))
        print(f"annotations reformatted -> {destination.as_posix()}")
