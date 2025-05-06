from typing import Annotated, Optional

import typer

from ls_helper.models.interface_models import (
    InterfaceData,
    ProjectVariableExtensions,
    FieldExtension,
)
from ls_helper.models.main_models import (
    get_project,
    platforms_overview,
    ProjectCreate,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectModel,
    ProjectViewCreate,
)
from ls_helper.project_mgmt import ProjectMgmt
from tools.project_logging import get_logger

logger = get_logger(__file__)

project_app = typer.Typer(
    name="Project setup", pretty_exceptions_show_locals=True
)


@project_app.command(short_help="Create a new project in LS", help="xxxx")
def create_project(
    title: Annotated[str, typer.Option()],
    alias: Annotated[str, typer.Option()],
    platform: Annotated[str, typer.Option()],
    language: Annotated[str, typer.Option()],
    maximum_annotations: Annotated[int, typer.Option()] = 2,
    create_coding_game_view: Annotated[bool, typer.Option()] = True,
):
    po = platforms_overview.create(
        ProjectCreate(
            title=title,
            platform=platform,
            language=language,
            alias=alias,
            default=False,
        )
    )

    values = ProjectMgmt.default_project_values()
    if maximum_annotations:
        values["maximum_annotations"] = maximum_annotations

    res = ls_client().patch_project(po.id, values)
    po.save_project_data(res)

    if create_coding_game_view:
        view = ProjectMgmt.create_view(
            ProjectViewCreate.model_validate(
                {"project": po.id, "data": {"title": "Coding Game"}}
            )
        )
        view_id = view.id
        po.coding_game_view_id = view_id
        # todo. save again.


@project_app.command(
    short_help="[setup] Just needs to be run once, for each new LS project"
)
def setup_project_settings(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    maximum_annotations: Annotated[int, typer.Option()] = 2,
):
    po = get_project(id, alias, platform, language)
    values = ProjectMgmt.default_project_values()
    if maximum_annotations:
        values["maximum_annotations"] = maximum_annotations
    # del values["color"]
    # print(values)
    res = ls_client().patch_project(po.id, values)
    po.save_project_data(res)
    if not res:
        print("error updating project settings")


@project_app.command(
    short_help="[setup] Required for annotation result processing. needs project-data"
)
def generate_variable_extensions_template(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
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

    if add_if_not_exists:
        # todo: validate the build with the xml in the project_data
        po.save_extensions(res_template)
    else:
        po.save_extensions(res_template, "alt")


@project_app.command(short_help="[maint]")
def download_project_data(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Argument()] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
) -> ProjectModel:
    po = get_project(id, alias, platform, language)
    project_data = ls_client().get_project(po.id)

    if not project_data:
        raise ValueError(f"No project found: {po.id}")
    po.save_project_data(project_data)
    return project_data
