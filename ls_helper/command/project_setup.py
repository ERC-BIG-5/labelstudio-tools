from typing import Annotated, Optional

import typer

from ls_helper.models.main_models import get_project, platforms_overview, ProjectCreate
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import ProjectModel
from ls_helper.project_mgmt import ProjectMgmt
from tools.project_logging import get_logger

logger = get_logger(__file__)

project_app = typer.Typer(name="Project setup", pretty_exceptions_show_locals=True)


@project_app.command(short_help="Create a new project in LS",
             help="xxxx")
def create_project(
        title: Annotated[str, typer.Option()],
        alias: Annotated[str, typer.Option()],
        platform: Annotated[str, typer.Option()],
        language: Annotated[str, typer.Option()]
):
    platforms_overview.create(
        ProjectCreate(
            title=title,
            platform=platform,
            language=language,
            alias=alias,
            default=False,
        )
    )



@project_app.command(
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


@project_app.command(short_help="[maint]")
def download_project_data(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Argument()] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
) -> ProjectModel:
    po = get_project(id, alias, platform, language)
    project_data = ls_client().get_project(po.id)

    if not project_data:
        raise ValueError(f"No project found: {po.id}")
    po.save_project_data(project_data)
    return project_data