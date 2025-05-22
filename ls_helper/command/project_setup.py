from typing import Annotated, Optional

import typer
from deepdiff import DeepDiff

from ls_helper.models.interface_models import (
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
)
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
    platforms_overview.create(
        ProjectCreate(
            title=title,
            platform=platform,
            language=language,
            alias=alias,
            default=False,
        ),
        create_coding_game_view,
        maximum_annotations,
    )


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

    # TODO TEST (change done at commit: 8d4a8ab8)
    interf_struct = po.raw_interface_struct
    data: dict[str, FieldExtension] = {}

    for field in interf_struct.inputs:
        data[field] = FieldExtension()
    for field in interf_struct.ordered_fields:
        data[field] = FieldExtension()

    res_template = ProjectVariableExtensions(extensions=data)

    try:
        existing = po.variable_extensions.model_dump(include={"extensions"})
        if overwrite_if_exists:
            logger.info("overwriting existing variable extensions")
            po.save_extensions(res_template)
        else:
            # TODO there should be a merge/update strategy
            compare_res_template = res_template.model_dump(
                include={"extensions"}
            )
            diff = DeepDiff(existing, compare_res_template, view="tree")
            all_parts = [
                "dictionary_item_added",
                "dictionary_item_removed",
                "values_changed",
                "type_changes",
            ]
            for part in all_parts:
                print(part)
                for change in diff.get(part, []):
                    # print(change.__dict__)
                    change_path = change.path(output_format="list")
                    print(change_path)
                    if part == "type_changes":
                        print(
                            f"existing: '{change.t1}', new value: '{change.t2}'"
                        )

    except ValueError:  # no existing...
        if add_if_not_exists:
            po.save_extensions(res_template)
        else:
            po.save_extensions(res_template, "alt")

        # todo: validate the build with the xml in the project_data


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
