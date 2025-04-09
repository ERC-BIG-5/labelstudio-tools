import json
from pathlib import Path
from typing import Annotated

import typer

from ls_helper.models.interface_models import InterfaceData, ProjectFieldsExtensions, FieldExtension
from ls_helper.new_models import ProjectAnnotationResultsModel, get_project
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

setup_app = typer.Typer(name="Setup projects", pretty_exceptions_show_locals=True)


@setup_app.command(short_help="[setup] Required for annotation result processing. needs project-data")
def generate_variable_extensions_template(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
):
    po = get_project(id, alias, platform, language)

    conf = po.interface

    def get_variable_extensions(annotation_struct: InterfaceData) -> ProjectFieldsExtensions:
        data: dict[str, FieldExtension] = {}

        for field in annotation_struct.inputs:
            data[field] = FieldExtension()
        for field in annotation_struct.ordered_fields:
            data[field] = FieldExtension()

        return ProjectFieldsExtensions(extensions= data)

    res_template = get_variable_extensions(conf)

    universal_fixes = read_data(SETTINGS.unifix_file_path)
    for k in res_template.extensions:
        if k in universal_fixes:
            # todo, can delete them?
            print(k)

    dest = SETTINGS.temp_file_path / f"result_fix_template_{po.id}.json"
    dest.write_text(res_template.model_dump_json())
    print(f"file -> {dest.as_posix()}")

@setup_app.command(short_help="[setup] ...")
def e():
    pass