from typing import Annotated

import typer

from ls_helper.models.interface_models import (
    InterfaceData,
    ProjectVariableExtensions,
    FieldExtension,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import get_project
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

setup_app = typer.Typer(name="Setup projects", pretty_exceptions_show_locals=True)


@setup_app.command(
    short_help="[setup] Required for annotation result processing. needs project-data"
)
def generate_variable_extensions_template(
    id: Annotated[int, typer.Option()] = None,
    alias: Annotated[str, typer.Option("-a")] = None,
    platform: Annotated[str, typer.Argument()] = None,
    language: Annotated[str, typer.Argument()] = None,
    add_if_not_exists: Annotated[bool, typer.Option()] = True,
    overwrite_if_exists: Annotated[bool, typer.Option()] = False,
):
    po = get_project(id, alias, platform, language)

    def get_variable_extensions(
        annotation_struct: InterfaceData,
    ) -> ProjectVariableExtensions:
        data: dict[str, FieldExtension] = {}

        for field in annotation_struct.inputs:
            data[field] = FieldExtension()
        for field in annotation_struct.ordered_fields:
            data[field] = FieldExtension()

        return ProjectVariableExtensions(extensions=data)

    res_template = get_variable_extensions(po.raw_interface_struct)

    universal_extensions = read_data(SETTINGS.unifix_extensions_file_path)

    filtered_ext = []
    for k in res_template.extensions:
        if k in universal_extensions:
            logger.info(f"taking {k} from universal extensions")
            continue
        filtered_ext.append(k)

    if add_if_not_exists:
        # todo: validate the build with the xml in the project_data
        po.save_extensions(res_template)
    else:
        po.save_extensions(res_template, "alt")


@setup_app.command(
    short_help="[setup] Just needs to be run once, for each new LS project"
)
def setup_project_settings(
    id: Annotated[int, typer.Option()] = None,
    alias: Annotated[str, typer.Option("-a")] = None,
    platform: Annotated[str, typer.Argument()] = None,
    language: Annotated[str, typer.Argument()] = None,
    maximum_annotations: Annotated[int, typer.Option()] = 2,
):
    po = get_project(id, alias, platform, language)
    values = ProjectMgmt.default_project_values()
    if maximum_annotations:
        values["maximum_annotations"] = maximum_annotations
    # del values["color"]
    print(values)
    res = ls_client().patch_project(po.id, values)
    po.save_project_data(res)
    if not res:
        print("error updating project settings")
