import json
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from ls_helper.agreements_calculation import Agreements
from ls_helper.annotation_timing import (
    annotation_total_over_time,
    plot_cumulative_annotations,
    get_annotation_lead_times,
    plot_date_distribution,
    annotation_timing,
)
from ls_helper.command.task import patch_tasks
from ls_helper.models.main_models import (
    get_project,
    get_p_access,
    ProjectAnnotationResultsModel,
)
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

annotations_app = typer.Typer(
    name="Annotations", pretty_exceptions_show_locals=True
)


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


@annotations_app.command(short_help="[stats] Annotation basic results")
def annotations(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    accepted_ann_age: Annotated[
        int, typer.Option(help="Download annotations if older than x hours")
    ] = 6,
    min_coders: Annotated[int, typer.Option()] = 2,
) -> tuple[Path, str]:
    po = get_project(id, alias, platform, language)
    po.validate_extensions()
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    # todo, this is not nice lookin ... lol
    res = mp.flatten_annotation_results(
        min_coders, mp.interface.ordered_fields
    )
    res = mp.format_df_for_csv(res)
    dest = SETTINGS.annotations_results_dir / f"{mp.id}.csv"
    res.to_csv(dest, index=False)
    print(f"annotation results -> {dest}")
    return dest


@annotations_app.command(
    short_help="[plot] Plot the completed tasks over time"
)
def status(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    accepted_ann_age: Annotated[
        int, typer.Option(help="Download annotations if older than x hours")
    ] = 6,
):
    po = get_project(id, alias, platform, language)
    _, project_annotations = ProjectMgmt.get_recent_annotations(
        po.id, accepted_ann_age
    )
    pa: Optional[ProjectAnnotationResultsModel] = project_annotations
    if project_annotations:
        df = annotation_timing(pa, po.project_data.maximum_annotations)
        temp_file = plot_date_distribution(df)
        open_image_simple(temp_file.name)
        temp_file.close()

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


@annotations_app.command(
    short_help="[plot] Plot the total completed tasks over day"
)
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


@annotations_app.command(
    short_help="[plot] Plot the total completed tasks over day"
)
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


@annotations_app.command(
    short_help="[stats] calculate general agreements stats"
)
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
    exclude_variables: Annotated[Optional[list[str]], typer.Argument()] = None,
) -> tuple[Path, Agreements]:
    """

    :param id:
    :param alias:
    :param platform:
    :param language:
    :param accepted_ann_age:
    :param max_num_coders:
    :param variables:
    :param exclude_variables: when no variables are selected, exclude those (from the all variables)
    :return:
    """
    dest, agreement = (
        get_project(id, alias, platform, language)
        .get_annotations_results(accepted_ann_age=accepted_ann_age)
        .get_coder_agreements(
            max_num_coders, variables, exclude_variables, True
        )
    )

    return dest, agreement


@annotations_app.command(short_help="Add conflicts to tasks data")
def add_conflicts_to_tasks(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
):
    po = get_project(id, alias, platform, language)
    # get conflicts and create the strings that are added to the tasks data
    conflicts = read_data(
        po.path_for(SETTINGS.agreements_dir, alternative=f"{po.id}_conflicts")
    )
    tasks: dict[int, set] = {}
    for variable, variable_data in conflicts.items():
        for option, var_option in variable_data.items():
            for task_id in var_option["conflict"]:
                task_conflicts = tasks.setdefault(task_id, set())
                task_conflicts.update(
                    [f"{variable};", f"{variable}@{option};"]
                )
    tasks_conflicts = {t: "".join(c) for t, c in tasks.items()}

    #
    po_tasks = po.get_tasks()
    for task in po_tasks.root:
        task.data["conflicts"] = tasks_conflicts.get(task.id, "")

    temp_file = po.path_for(SETTINGS.temp_file_path)
    temp_file.write_text(
        json.dumps(
            [
                t.model_dump(include={"data", "id", "project"})
                for t in po_tasks.root
            ],
            indent=2,
        )
    )
    # pass
    patch_tasks(temp_file, po.id)
