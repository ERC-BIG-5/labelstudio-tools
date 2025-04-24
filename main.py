import json
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer
from deepdiff import DeepDiff
from tqdm import tqdm

from ls_helper.annotation_timing import (
    annotation_total_over_time,
    get_annotation_lead_times,
    plot_cumulative_annotations,
    plot_date_distribution,
)
from ls_helper.command.annotations import annotations_app
from ls_helper.command.backup import backup_app
from ls_helper.command.labeling_conf import labeling_conf_app
from ls_helper.command.pipeline import pipeline_app
from ls_helper.command.project_setup import project_app
from ls_helper.command.setup import (
    setup_app,
)
from ls_helper.command.task import task_add_predictions, task_app
from ls_helper.command.view import view_app
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.fresh_agreements import Agreements
from ls_helper.funcs import build_view_with_filter_p_ids
from ls_helper.models.interface_models import IChoices
from ls_helper.models.main_models import (
    get_p_access,
    get_project,
    platforms_overview,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectViewCreate,
    ProjectViewDataModel,
    ProjectViewModel,
)
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(
    name="Labelstudio helper", pretty_exceptions_show_locals=False
)

app.add_typer(setup_app, name="setup", short_help="Commands related to initializing the project")
app.add_typer(project_app, name="project", short_help="Commands related to project setup and maintenance")
app.add_typer(backup_app, name="backup", short_help="Commands related to bulkbacking up projects and annotations")
app.add_typer(labeling_conf_app, name="labeling-conf",
              short_help="Commands related to building, validating and uploading project label configurations")
app.add_typer(task_app, name="task", short_help="Commands related to downloading, creating and patching project tasks")
app.add_typer(view_app, name="view", short_help="Commands related to project views")
app.add_typer(annotations_app, name="annotations",
              short_help="Commands related to downloading and analyzing annotations")
app.add_typer(pipeline_app, name="pipeline", short_help="Commands related to interaction with the Pipeline package")


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


@app.command(
    short_help="[ls maint] Update tasks. Files must be matching lists of {id: , data:}"
)
# todo: more testing
def strict_update_project_tasks(
        new_data_file: Path, existing_data_file: Optional[Path] = None
):
    raise NotImplementedError("client.patch_task parameters changed")
    client = ls_client()
    new_data_list = json.loads(new_data_file.read_text(encoding="utf-8"))
    if existing_data_file:
        existing_data_list = json.loads(
            existing_data_file.read_text(encoding="utf-8")
        )
        assert len(new_data_list) == len(existing_data_list)

        for idx, t in tqdm(enumerate(new_data_list)):
            t_id = t["id"]
            ex_t = existing_data_list[idx]
            assert t_id == ex_t["id"]
            strict_update_project_task_data(t_id, t, ex_t)

        print(f"{len(new_data_list)} tasks updated")
        return

    for t in tqdm(new_data_list):
        client.patch_task(t["id"], t["data"])

    print(f"{len(new_data_list)} tasks updated")


@app.command(short_help="[maint]")
def download_project_views(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
) -> list[ProjectViewModel]:
    p_a = get_p_access(id, alias, platform, language)
    po = get_project(p_a)
    views = ProjectMgmt.refresh_views(po)
    logger.debug(f"view file -> {po.view_file}")
    return views


@app.command(short_help="[plot] Plot the completed tasks over time")
def status(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
):
    from ls_helper import main_funcs

    po = get_project(id, alias, platform, language)
    main_funcs.status(po, accepted_ann_age)

    """ experiment. redo nicer. getting count per user
    po = get_project(id, alias, platform, language)
    po.validate_extensions()
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    # todo, this is not nice lookin ... lol
    _ = mp.basic_flatten_results(1)
    # just for checking...
    #client = ls_client()
    #users = client.get_users()
    #fix_users(res, {u.id: u.username for u in users})
    #print(res["user_id"].value_counts())
    """


@app.command(short_help="[plot] Plot the total completed tasks over day")
def total_over_time(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
):
    print(get_p_access(id, alias, platform, language))
    po = get_project(id, alias, platform, language)
    annotations = ProjectMgmt.get_recent_annotations(po.id, accepted_ann_age)
    df = annotation_total_over_time(annotations)
    temp_file = plot_cumulative_annotations(
        df, f"{po.platform}/{po.language}: Cumulative Annotations Over Time"
    )
    dest = SETTINGS.plots_dir / f"{platform}-{language}.png"
    shutil.copy(temp_file.title, dest)
    temp_file.close()
    open_image_simple(dest)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def annotation_lead_times(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
):
    po = get_project(id, alias, platform, language)
    project_annotations = ProjectMgmt.get_recent_annotations(
        po.id, accepted_ann_age
    )

    df = get_annotation_lead_times(project_annotations)
    temp_file = plot_date_distribution(df, y_col="lead_time")

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="[stats] calculate general agreements stats")
def agreements(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
        max_num_coders: Annotated[int, typer.Option()] = 2,
        variables: Annotated[Optional[list[str]], typer.Argument()] = None,
) -> tuple[Path, Agreements]:
    """

    :param id:
    :param alias:
    :param platform:
    :param language:
    :param accepted_ann_age:
    :param max_num_coders:
    :param variables:
    :return:
    """
    dest, agreement = (
        get_project(id, alias, platform, language)
        .get_annotations_results(accepted_ann_age=accepted_ann_age)
        .get_coder_agreements(max_num_coders, variables, True)
    )

    return dest, agreement


def get_all_variable_names(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    # todo redo and test...
    struct = po.raw_interface_struct
    return list(struct.ordered_fields_map.keys())


def get_variables_info(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    return [
        {
            "name": k,
            "required": v.required,
            "choice_type": v.choice if isinstance(v, IChoices) else None,
        }
        for k, v in po.raw_interface_struct.ordered_fields_map.items()
    ]


@app.command(short_help="create or update a view for variable conflict")
def create_conflict_view(
        variable: Annotated[str, typer.Option()],
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
        variable_option: Annotated[Optional[str], typer.Option()] = None,
):
    # todo redo with fresh_agreement
    """
    po = get_project(id, alias, platform, language)
    # po.validate_extensions()
    # mp = po.get_annotations_results()

    # just check existence
    # if not po.interface.ordered_fields_map.get(variable):
    #    raise ValueError(f"Variable {variable} has not been defined")
    # agreement_data = json.loads((SETTINGS.agreements_dir / f"{po.id}.json").read_text())

    agg_metrics = po.get_agreement_data().agreement_metrics
    if variable.endswith("_visual"):
        variable = variable + "_0"
    print(variable)
    am = agg_metrics.all_variables.get(variable)
    if not am or not len(am.conflicts):
        all_c = po.get_agreement_data().conflicts

        dd_t = []
        for c in all_c:
            # print(c)
            if c.variable == variable:
                dd_t.append(c)
            task_ids = [c.task_id for c in dd_t][:50]
    else:
        # for the broken version, single are strings of task-id and _0
        if isinstance(am, SingleChoiceAgreement):
            task_ids = [int(s.split("_")[0]) for s in am.conflicts][:30]

        else:
            task_ids = [int(str(s.task_id)[:-1]) for s in am.conflicts][:30]

    title = f"conflict:{variable}"
    view = ProjectMgmt.create_view(ProjectViewCreate.model_validate({"project": po.id, "data": {
        "title": title,
        "filters": build_platform_id_filter(task_ids, "task_id")}}))
    url = f"{SETTINGS.LS_HOSTNAME}/projects/{po.id}/data?tab={view.id}"
    print(url)
    return url
    """


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


def add_prediction_test():
    resp = task_add_predictions(
        33030,
        {
            "model_version": "one",
            "score": 0.5,
            # "type": "choices",
            "result": [
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_any",
                    "value": {"choices": ["Yes"]},
                },
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_visual",
                    "value": {"choices": ["Yes"]},
                },
            ],
        },
    )
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    twitter = "twitter"
    youtube = "youtube"
    en = "en"
    es = "es"
    tw_es = {"platform": twitter, "language": es}
    yt_en4 = {"id": 50}
    _default = tw_es

    # setup
    from ls_helper.command import setup

    setup.add_projects()

    # this will work, since there is just one spanish twitter (so its set to default)
    agreements(**{"alias": "twitter-en-3"}, variables=["nature_any"])
