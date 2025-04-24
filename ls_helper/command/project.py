import typer

from ls_helper.models.main_models import get_project
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.project_mgmt import ProjectMgmt
from tools.project_logging import get_logger
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tools.project_logging import get_logger
from tqdm import tqdm

logger = get_logger(__file__)

project_setup_app = typer.Typer(name="Project setup", pretty_exceptions_show_locals=True)


@project_setup_app.command(
    short_help="[setup] Just needs to be run once, for each new LS project"
)
def setup_project_settings(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        maximum_annotations: Annotated[int, typer.Option()] = 2,
):
    po = get_project(id, alias, platform, language)
    values = ProjectMgmt.default_project_values()
    if maximum_annotations:
        values["maximum_annotations"] = maximum_annotations
    # del values["color"]
    # print(values)
    res = ls_client().patch_project(po.id, values)
    po.save_project_data(res)
    if not res:
        print("error updating project settings")
