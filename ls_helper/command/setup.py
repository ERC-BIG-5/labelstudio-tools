from typing import Annotated, Optional

import typer
from tqdm import tqdm

from tools.files import read_data
from tools.project_logging import get_logger

from ls_helper.models.interface_models import (
    FieldExtension,
    InterfaceData,
    ProjectVariableExtensions,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import get_project, ProjectData, ProjectOverview
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS

logger = get_logger(__file__)

setup_app = typer.Typer(
    name="Setup projects", pretty_exceptions_show_locals=True
)


@setup_app.command(
    short_help="Create the projects.json file",
)
def add_projects():
    projects = [
        ProjectData.model_validate(
            {"id": p.id, "alias": str(p.id), "title": p.title}
        )
        for p in ls_client().projects_list()
    ]
    overview = ProjectOverview.load()
    stored_projects = overview.project_list()
    stored_projects_ids = [p.id for p in stored_projects]
    added: list[ProjectData] = []
    for p in projects:
        if p.id not in stored_projects_ids:
            if p.alias in overview.alias_map:
                logger.warning(f"Cannot add project {repr(p)} because alias: {p.alias} exists already in overview file")
                continue
            added.append(p)
            overview.projects[p.id] = p
    # in case program doesn't quit
    overview.create_map()
    overview.save()
    logger.info(f"Added the projects:\n {'\n '.join([repr(p) for p in added])}\n"
                f"You might want to edit {SETTINGS.projects_main_file} and replace the aliases, platform and language values.")


@setup_app.command(
    short_help="Download all LS project data"
)
def download_all_projects():
    for project in tqdm(ProjectOverview.load().project_list()):
        project_data = ls_client().get_project(project.id)

        if not project_data:
            raise ValueError(f"No project found: {project.id}")
        project.save_project_data(project_data)

@setup_app.command(
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
    print(values)
    res = ls_client().patch_project(po.id, values)
    po.save_project_data(res)
    if not res:
        print("error updating project settings")
