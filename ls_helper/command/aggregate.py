from typing import Annotated

import typer

from ls_helper.models.main_models import get_project, ProjectData
from tools.project_logging import get_logger

logger = get_logger(__file__)

aggregate_app = typer.Typer(
    name="Aggregate",
    short_help="Aggregate results over several projects",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


@aggregate_app.command(short_help="")
def check_projects(
    id: Annotated[list[int], typer.Option(help="")] = None,
) -> None:
    projects: list[ProjectData] = [
        get_project(project_id) for project_id in id
    ]
    for p in projects:
        print(p)


@aggregate_app.command(short_help="")
def do_something_else(id: Annotated[list[int], typer.Option()] = None) -> None:
    pass
