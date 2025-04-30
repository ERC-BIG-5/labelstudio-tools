from typing import Annotated, Optional

import typer
from tqdm import tqdm

from tools.files import read_data
from tools.project_logging import get_logger

from ls_helper.models.interface_models import (
    FieldExtension,
    InterfaceData,
    ProjectVariableExtensions,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.models.main_models import (
    get_project,
    ProjectData,
    ProjectOverview,
)
from ls_helper.settings import SETTINGS

logger = get_logger(__file__)

setup_app = typer.Typer(
    name="Setup projects", pretty_exceptions_show_locals=True
)


@setup_app.command(
    short_help="Create the projects.json file",
)
def add_projects():
    projects = [
        ProjectData.model_validate(
            {"id": p.id, "alias": str(p.id), "title": p.title}
        )
        for p in ls_client().projects_list()
    ]
    overview = ProjectOverview.load()
    stored_projects = overview.project_list()
    stored_projects_ids = [p.id for p in stored_projects]
    added: list[ProjectData] = []
    for p in projects:
        if p.id not in stored_projects_ids:
            if p.alias in overview.alias_map:
                logger.warning(
                    f"Cannot add project {repr(p)} because alias: {p.alias} exists already in overview file"
                )
                continue
            added.append(p)
            overview.projects[p.id] = p
    # in case program doesn't quit
    overview.create_map()
    overview.save()
    logger.info(
        f"Added the projects:\n {'\n '.join([repr(p) for p in added])}\n"
        f"You might want to edit {SETTINGS.projects_main_file} and replace the aliases, platform and language values."
    )


@setup_app.command(short_help="Download all LS project data")
def download_all_projects():
    for project in tqdm(ProjectOverview.load().project_list()):
        project_data = ls_client().get_project(project.id)

        if not project_data:
            raise ValueError(f"No project found: {project.id}")
        project.save_project_data(project_data)

