from pathlib import Path
from typing import Annotated, Optional

import typer
from deepdiff import DeepDiff

from ls_helper.config_helper import parse_label_config_xml
from ls_helper.settings import SETTINGS
from main import app
from tools.project_logging import get_logger

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.models.main_models import get_project, platforms_overview

logger = get_logger(__file__)

labeling_conf_app = typer.Typer(
    name="Labeling config", pretty_exceptions_show_locals=True
)

# todo duplicate in main


@labeling_conf_app.command(short_help="Create labeling config from template",
                           help="Uses platform specific template")
def build_ls_labeling_interface(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("--alias","-a")] = None,
    platform: Annotated[Optional[str], typer.Option()] = None,
    language: Annotated[Optional[str], typer.Option()] = None,
    alternative_template: Annotated[Optional[str], typer.Argument()] = None,
) -> Optional[Path]:
    po = get_project(id, alias, platform, language)
    path, tree = po.build_ls_labeling_config(alternative_template)
    return path


@labeling_conf_app.command(help="[ls maint] Upload labeling config")
def update_labeling_config(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    alternative_built: Annotated[Optional[str], typer.Argument()] = None,
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
        print(
            f"Could not update labeling config for {platform}/{language}/{id}"
        )
        return
    pass
    po.save_project_data(project)
    # dest = SETTINGS.projects_dir / f"{p.id}.json"
    # dest.write_text(project_data.model_dump_json())

    print(f"updated labeling config for {po.platform}/{po.language}/{po.id}")


@app.command()
def build_extension_index(
        take_all_defaults: Annotated[
            bool, typer.Option(help="take default projects (pl/lang)")
        ] = True,
        project_ids: Annotated[Optional[list[int]], typer.Option("-pid")] = None,
):
    """
    Checks
    :param take_all_defaults:
    :param project_ids:
    :return:
    """
    from ls_helper.annot_extension import (
        build_extension_index as _build_extension_index,
    )

    if project_ids:
        projects = [get_project(id) for id in project_ids]
    elif take_all_defaults:
        projects = list(platforms_overview.default_map.values())
    else:
        raise ValueError("Unclear parameter for build_extension_index")
    index = _build_extension_index(projects)
    dest = (
            SETTINGS.temp_file_path
            / f"annot_ext_index_{'_'.join(str(p.id) for p in projects)}.json"
    )
    dest.write_text(index.model_dump_json(indent=2))
    print(f"index saved to {dest}")


@app.command()
def check_labelling_config(
        build_file_name: str,
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)

    existing_struct = po.raw_interface_struct

    if not build_file_name.endswith(".xml"):
        build_file_name += ".xml"
    fp = SETTINGS.built_labeling_configs / build_file_name

    new_config = parse_label_config_xml(fp.read_text())
    print(type(new_config), type(existing_struct))
    diff = DeepDiff(new_config, existing_struct)
    print(diff.to_json(indent=2))
    pass
