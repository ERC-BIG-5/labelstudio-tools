from datetime import datetime, timedelta
from typing import Annotated, Optional

import typer
import webbrowser
import pathlib

from ls_helper.annotation_timing import annotation_timing, plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.funcs import get_latest_annotation, get_latest_annotation_file
from ls_helper.models import ProjectAnnotations
from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS

app = typer.Typer(name="Labelstudio helper")


def ls_client() -> LabelStudioBase:
    return LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = pathlib.Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


def get_recent_annotations(project_id: int, accepted_age: int) -> Optional[ProjectAnnotations]:
    latest_file = get_latest_annotation_file(project_id)
    stem = latest_file.stem
    date_part = stem.split("-")[1]

    file_dt = datetime.strptime(date_part, "%Y%m%d_%H%M")
    # print(file_dt, datetime.now(), datetime.now() - file_dt)
    if datetime.now() - file_dt > timedelta(hours=accepted_age):
        print("downloading annotations")
        return ls_client().get_project_annotations(project_id)
    else:
        return get_latest_annotation(project_id)


@app.command(short_help="Plot the completed tasks over time")
def status(project_id: Annotated[int, typer.Option()],
           accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 6):
    project_annotations = get_recent_annotations(project_id, accepted_ann_age)

    df = annotation_timing(project_annotations)
    temp_file = plot_date_distribution(df)

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="Plot the total completed tasks over day")
def annotation_lead_times(project_id: Annotated[int, typer.Option()],
                          accepted_ann_age: Annotated[
                              int, typer.Option(help="Download annotations if older than x hours")] = 6):
    project_annotations = get_recent_annotations(project_id, accepted_ann_age)

    df = get_annotation_lead_times(project_annotations)
    temp_file = plot_date_distribution(df, y_col="lead_time")

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="Plot the total completed tasks over day")
def total_over_time(project_id: Annotated[int, typer.Option()],
                    accepted_ann_age: Annotated[
                        int, typer.Option(help="Download annotations if older than x hours")] = 6):
    df = annotation_total_over_time(get_recent_annotations(project_id, accepted_ann_age))
    temp_file = plot_cumulative_annotations(df)

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command()
def clean_project_task_files(project_id: Annotated[int, typer.Option()],
                             title: Optional[str] = None, ):
    pass
    # 1. get project_sync folder
    # curl http://localhost:8080/api/storages/localfiles/ \
    #      -H "Authorization: Token  <api_key>"
    # query-Param: project
    # 2. get project tasks
    # remove all files that are not in a task
    client = ls_client()

    resp = client.list_import_storages(project_id)
    local_storages = resp.json()

    if len(local_storages) == 0:
        print("No storages found")
        return
    if len(local_storages) > 1:
        if not title:
            print("Multiple storages found.provide the 'title'")
            return
        lc = [lc for lc in local_storages if lc["title"] == title]
        if len(lc) == 0:
            print(f"No storages found with title: {title}")
            return
        lc = lc[0]
    else:
        lc = local_storages[0]
    path = pathlib.Path(lc["path"])

    rel_path = path.relative_to(SETTINGS.IN_CONTAINER_LOCAL_STORAGE_BASE)
    host_path = SETTINGS.HOST_STORAGE_BASE / rel_path

    existing_task_files = list(host_path.glob("*.json"))

    # print(host_path.absolute())

    resp = client.get_task_list(project=project_id)
    tasks = resp.json()["tasks"]
    used_task_files = [task.get("storage_filename") for task in tasks]
    # filter Nones
    used_task_files = [t for t in used_task_files if t]
    obsolete_files = set(existing_task_files) - set(used_task_files)
    print([o.relative_to(host_path) for o in obsolete_files])


# console = Console()

if __name__ == "__main__":
    # status(33)
    # p = client.get_project(31)
    # print(p)
    # res = client.patch_project(31, {"is_draft": True, "is_published": True})
    # print(res)
    # ProjectMgmt.update_projects()
    # print(ProjectMgmt.get_annotations("youtube","en"))
    clean_project_task_files(33)
