from typing import Annotated, Optional

import typer
from tools.project_logging import get_logger
from tqdm import tqdm

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import ProjectData, platforms_overview

logger = get_logger(__file__)

backup_app = typer.Typer(
    name="Make Backups", pretty_exceptions_show_locals=True
)


@backup_app.command(short_help="[stats] Annotation basic results")
def backup(
    objects: Annotated[Optional[list[str]], typer.Option()] = None,
    dl_all_projects: Annotated[Optional[bool], typer.Option()] = None,
) -> None:
    if dl_all_projects:
        projects_ = ls_client().projects_list()
        projects = [
            ProjectData.model_validate(
                {"id": p.id, "alias": str(p.id), "title": p.title}
            )
            for p in projects_
        ]
    else:
        projects = list(platforms_overview.projects.values())
    for project in tqdm(projects):
        print(project.alias)
        project.fetch_annotations()
        # project.get_tasks()


if __name__ == "__main__":
    backup_app()
