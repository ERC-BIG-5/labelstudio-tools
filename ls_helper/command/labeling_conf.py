from pathlib import Path
from typing import Annotated, Optional

import typer
from deepdiff import DeepDiff

from ls_helper.command.project_setup import download_project_data
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.models.main_models import get_project, platforms_overview
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.settings import SETTINGS
from tools.project_logging import get_logger

logger = get_logger(__file__)

labeling_conf_app = typer.Typer(
    name="Labeling config", pretty_exceptions_show_locals=True
)


@labeling_conf_app.command(
    short_help="Create labeling config from template",
    help="Uses platform specific template",
)
def build(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("--alias", "-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
        alternative_template: Annotated[Optional[str], typer.Argument()] = None,
) -> tuple[Path, bool]:
    """

    :param id:
    :param alias:
    :param platform:
    :param language:
    :param alternative_template:
    :return:  path, is_valid
    """
    po = get_project(id, alias, platform, language)
    path, tree, valid = po.label_config.build_ls_labeling_config(alternative_template)
    return path, valid


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

    label_config = po.label_config.read_built_labeling_config(alternative_built)

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


@labeling_conf_app.command()
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


@labeling_conf_app.command()
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


@labeling_conf_app.command()
def from_project(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    """
    Get the labeling conf from a concrete project file and dump it to labeling_configs/project/<id>.xml
    :param id:
    :param alias:
    :param platform:
    :param language:
    :return:
    """
    download_project_data(id, alias, platform, language)
    po = get_project(id, alias, platform, language)
    config_xml = po.project_data.label_config
    dest_file = (
            SETTINGS.BASE_DATA_DIR / f"labeling_configs/project/{po.id}.xml"
    )
    dest_file.parent.mkdir(exist_ok=True)
    dest_file.write_text(config_xml)
    print(dest_file)


def patch_from_modyfied_project_xml(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    """
    get the xml from 'labeling_configs/project/{po.id}.xml'
    :param id:
    :param alias:
    :param platform:
    :param language:
    :return:
    """
    pass
