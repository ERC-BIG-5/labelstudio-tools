from pathlib import Path
from typing import Annotated, Optional, Callable

import typer

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import platforms_overview, ProjectData
from tools.project_logging import get_logger

logger = get_logger(__file__)

labeling_conf_app = typer.Typer(name="Labeling config", pretty_exceptions_show_locals=True)
app = labeling_conf_app

# todo duplicate in main
_project: Callable = lambda id, al, pl, la: platforms_overview.get_project((id, al, pl, la))


@app.command()
def build_ls_labeling_interface(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Option()] = None,
        language: Annotated[str, typer.Option()] = None,
        alternative_template: Annotated[str, typer.Argument()] = None,
) -> Optional[Path]:
    po = _project(id, alias, platform, language)
    path, tree = po.build_ls_labeling_config(alternative_template)
    return path



@app.command(help="[ls maint] Upload labeling config")
def update_labeling_config(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
):
    # todo, if we do that. save it
    # download_project_data(platform, language)
    client = ls_client()
    po = _project(id, platform, language, alias)

    label_config = po.read_labeling_config()

    print("Validating config")
    resp = client.validate_project_labeling_config(id, label_config)
    if resp.status_code != 200:
        print("not valud")
        print(resp.json())
        return
    res = client.patch_project(id, {"label_config": label_config})
    if not res:
        print(f"Could not update labeling config for {platform}/{language}/{id}")
        return

    # po.save_project_data()
    # dest = SETTINGS.projects_dir / f"{p.id}.json"
    # dest.write_text(project_data.model_dump_json())

    print(f"updated labeling config for {platform}/{language}/{id}")
