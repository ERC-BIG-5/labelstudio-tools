import json
import shutil
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

import typer

from ls_helper.annotation_timing import annotation_timing, plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.funcs import get_latest_annotation, get_latest_annotation_file, build_view_with_filter_p_ids
from ls_helper.models import ProjectAnnotations, ProjectOverview
from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import ProjectViewModel
from ls_helper.settings import SETTINGS

app = typer.Typer(name="Labelstudio helper")


def ls_client() -> LabelStudioBase:
    return LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
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
                             title: Annotated[Optional[str], typer.Option()] = None,
                             just_check: Annotated[bool, typer.Option()] = False):
    pass
    # 1. get project_sync folder
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
    path = Path(lc["path"])

    rel_path = path.relative_to(SETTINGS.IN_CONTAINER_LOCAL_STORAGE_BASE)
    host_path = SETTINGS.HOST_STORAGE_BASE / rel_path

    existing_task_files = list(host_path.glob("*.json"))
    existing_task_files = [f.name for f in existing_task_files]
    # print(existing_task_files)
    # print("**************")
    # print(host_path.absolute())
    print("getting task  list...")
    resp = client.get_task_list(project=project_id)
    tasks = resp.json()["tasks"]
    used_task_files = [task.get("storage_filename") for task in tasks]
    # filter Nones
    used_task_files = [Path(t) for t in used_task_files if t]
    used_task_files = [t.name for t in used_task_files]
    # print(used_task_files)

    obsolete_files = set(existing_task_files) - set(used_task_files)

    # print([o.relative_to(host_path) for o in obsolete_files])
    # json.dump(list(obsolete_files), Path("t.json").open("w"))
    if just_check:
        print(f"{len(obsolete_files)} would be moved")
        return
    print(f"{len(obsolete_files)} will be moved")

    backup_dir = SETTINGS.DELETED_TASK_FILES_BACKUP_BASE_DIR
    backup_final_dir = backup_dir / str(project_id)
    backup_final_dir.mkdir(parents=True, exist_ok=True)
    for f in obsolete_files:
        src = host_path / f
        # print(src.exists())
        shutil.move(src, backup_final_dir / f)


@app.command()
def download_project_views(platform: Annotated[str, typer.Option()], language: Annotated[str, typer.Option()]) -> list[
    ProjectViewModel]:
    client = ls_client()
    po = ProjectOverview.projects()
    project_id = po.get_project_id(platform, language)
    views = client.get_project_views(project_id)
    dest = po.get_view_file(project_id)
    dest.write_text(json.dumps([v.model_dump() for v in views]))
    return views


@app.command()
def set_view_items(platform: Annotated[str, typer.Option()],
                   language: Annotated[str, typer.Option()],
                   view_title: Annotated[str, typer.Option()],
                   project_id_file: Annotated[Path, typer.Option()]):
    po = ProjectOverview.projects()
    views = po.get_views((platform, language))
    if not views:
        print("No views found")
        return
    _view: ProjectViewModel = None
    for view in views:
        if view.data.title == view_title:
            _view = view
            break
    if not _view:
        views_titles = [v.data.title for v in views]
        print(f"No views found: '{view_title}', candidates: {views_titles}")
        return
    # check the file:
    if not project_id_file.exists():
        print(f"file not found: {project_id_file}")
        return
    project_ids = json.load(project_id_file.open())
    assert isinstance(project_ids, list)
    build_view_with_filter_p_ids(SETTINGS.client, _view, project_ids)
    print("View successfully updated")


if __name__ == "__main__":
    # status(33)
    # p = client.get_project(31)
    # print(p)
    # res = client.patch_project(31, {"is_draft": True, "is_published": True})
    # print(res)
    # ProjectMgmt.update_projects()
    # print(ProjectMgmt.get_annotations("youtube","en"))
    ## TODO
    clean_project_task_files(33)
    ##

    set_view_items("youtube", "en", "Old-Sentiment/Framing",
                   Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/temp/yt_en_problematic_tasks.json"))
