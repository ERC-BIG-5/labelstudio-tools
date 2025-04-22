from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.new_models import platforms_overview
from tools.project_logging import get_logger

logger = get_logger(__file__)

backup_app = typer.Typer(name="Make Backups", pretty_exceptions_show_locals=True)

@backup_app.command(short_help="[stats] Annotation basic results")
def backup(objects: Annotated[Optional[list[str]], typer.Option()]=None):
    for platform in tqdm(platforms_overview.projects.values()):
        print(platform.alias)
        platform.fetch_annotations()
        platform.get_tasks()

if __name__ == "__main__":
    backup_app()
