import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from ls_helper.annotation_timing import annotation_total_over_time, plot_cumulative_annotations, \
    get_annotation_lead_times, plot_date_distribution
from ls_helper.fresh_agreements import Agreements
from ls_helper.project_mgmt import ProjectMgmt

from tools.project_logging import get_logger

from ls_helper.models.main_models import get_project, get_p_access
from ls_helper.settings import SETTINGS

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


@annotations_app.command(short_help="[setup] ...")
def e():
    pass


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
