from pathlib import Path
from typing import Annotated, Optional

import typer

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import get_project
from tools.project_logging import get_logger

logger = get_logger(__file__)

labeling_conf_app = typer.Typer(
    name="Labeling config", pretty_exceptions_show_locals=True
)

# todo duplicate in main


@labeling_conf_app.command()
def build_ls_labeling_interface(
    id: Annotated[int, typer.Option()] = None,
    alias: Annotated[str, typer.Option("-a")] = None,
    platform: Annotated[str, typer.Option()] = None,
    language: Annotated[str, typer.Option()] = None,
    alternative_template: Annotated[str, typer.Argument()] = None,
) -> Optional[Path]:
    po = get_project(id, alias, platform, language)
    path, tree = po.build_ls_labeling_config(alternative_template)
    return path


@labeling_conf_app.command(help="[ls maint] Upload labeling config")
def update_labeling_config(
    id: Annotated[int, typer.Option()] = None,
    alias: Annotated[str, typer.Option("-a")] = None,
    platform: Annotated[str, typer.Argument()] = None,
    language: Annotated[str, typer.Argument()] = None,
    alternative_built: Annotated[str, typer.Argument()] = None,
):
    # todo, if we do that. save it
    # download_project_data(platform, language)
    client = ls_client()
    po = get_project(id, alias, platform, language)

    label_config = po.read_labeling_config(alternative_built)

    print("Validating config")
    resp = client.validate_project_labeling_config(po.id, label_config)
    if resp.status_code != 200:
        print("labeling config is not valid")
        print(resp.json())
        return

    project = client.patch_project(po.id, {"label_config": label_config})
    if not project:
        print(f"Could not update labeling config for {platform}/{language}/{id}")
        return
    pass
    po.save_project_data(project)
    # dest = SETTINGS.projects_dir / f"{p.id}.json"
    # dest.write_text(project_data.model_dump_json())

    print(f"updated labeling config for {po.platform}/{po.language}/{po.id}")
