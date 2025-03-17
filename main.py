import json
import shutil
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from irrCAC.raw import CAC

from ls_helper.ana_res import parse_label_config_xml
from ls_helper.annotation_timing import annotation_timing, plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.funcs import get_latest_annotation, get_latest_annotation_file, build_view_with_filter_p_ids
from ls_helper.models import ProjectAnnotations, ProjectOverview, MyProject, ProjectAnnotationExtension
from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import ProjectViewModel
from ls_helper.settings import SETTINGS, ls_logger

app = typer.Typer(name="Labelstudio helper")


def ls_client() -> LabelStudioBase:
    return LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


def get_recent_annotations(project_id: int, accepted_age: int) -> Optional[ProjectAnnotations]:
    latest_file = get_latest_annotation_file(project_id)
    if latest_file is not None:
        file_dt = datetime.strptime(latest_file.stem, "%Y%m%d_%H%M")
        # print(file_dt, datetime.now(), datetime.now() - file_dt)
        if datetime.now() - file_dt < timedelta(hours=accepted_age):
            ls_logger.info("Get recent, gets latest annotation")
            return get_latest_annotation(project_id)

    print("downloading annotations")
    return ls_client().get_project_annotations(project_id)


@app.command(short_help="Plot the completed tasks over time")
def status(platform: Annotated[str, typer.Argument()],
           language: Annotated[str, typer.Argument()],
           accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 6):
    po = ProjectOverview.projects().get_project((platform, language))
    project_annotations = get_recent_annotations(po.id, accepted_ann_age)

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


@app.command(short_help="Annotation basic results")
def annotations_results(platform: Annotated[str, typer.Option()],
                        language: Annotated[str, typer.Option()],
                        accepted_ann_age: Annotated[
                            int, typer.Option(help="Download annotations if older than x hours")] = 6,
                        min_coders: Annotated[int, typer.Option()] = 2) -> tuple[
    Path, str]:
    project_data = ProjectOverview.project_data(platform, language)
    if not project_data:
        print(ProjectOverview.projects())
        raise ValueError(f"No project data for {platform}/{language}")
    project_id = project_data["id"]

    conf = parse_label_config_xml(project_data["label_config"],
                                  project_id=project_id,
                                  include_text=True)

    annotations = get_recent_annotations(project_id, accepted_ann_age)

    data_extensions = None
    if (fi := SETTINGS.BASE_DATA_DIR / f"fixes/{project_id}.json").exists():
        data_extensions = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
    mp = MyProject(project_data=project_data,
                   annotation_structure=conf,
                   data_extensions=data_extensions,
                   raw_annotation_result=annotations)
    mp.calculate_results()
    mp.apply_extension(fillin_defaults=True)
    dest = SETTINGS.annotations_results_dir / f"{str(project_id)}.csv"
    mp.results2csv(dest, with_defaults=True, min_coders=min_coders)
    print(f"annotation results -> {dest}")
    return dest, annotations.file_path.stem


@app.command(short_help="Plot the total completed tasks over day")
def total_over_time(platform: Annotated[str, typer.Argument()],
                    language: Annotated[str, typer.Argument()],
                    accepted_ann_age: Annotated[
                        int, typer.Option(help="Download annotations if older than x hours")] = 6):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]
    df = annotation_total_over_time(get_recent_annotations(project_id, accepted_ann_age))
    temp_file = plot_cumulative_annotations(df)

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="delete the json files from the local storage folder, from tasks that habe been deleted")
def clean_project_task_files(project_id: Annotated[int, typer.Option()],
                             title: Annotated[Optional[str], typer.Option()] = None,
                             just_check: Annotated[bool, typer.Option()] = False):
    # 1. get project_sync folder
    # 2. get project tasks
    # remove all files that are not in a task
    """
    ON THE VM:
    sudo env PYTHONPATH=.  /home/ubuntu/projects/big5/platform_clients/.venv/bin/typer main.py run clean-project-task-files ...
    """

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
def download_project_data(
        platform: Annotated[str, typer.Argument()],
        language: Annotated[str, typer.Argument()]
):
    project_id = ProjectOverview.projects().get_project_id(platform, language)
    project_data = ls_client().get_project(project_id)
    if not project_data:
        raise ValueError(f"No project found: {project_id}")
    else:
        _dir = SETTINGS.projects_dir / platform
        _dir.mkdir(parents=True, exist_ok=True)
        dest = _dir / f"{language}.json"
        dest.write_text(project_data.model_dump_json())


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


@app.command()
def update_coding_game(platform: str, language: str) -> Optional[tuple[int, int]]:
    """
    if successful sends back project_id, view_id

    """

    po = ProjectOverview.projects().get_project((platform, language))
    view_id = po.coding_game_view_id
    if not view_id:
        print("No views found for coding game")
        return

    views = po.get_views()
    if not views:
        download_project_views(platform, language)
        views = po.get_views()
        # print("No views found for project. Call 'download_project_views' first")
        # return
    view_ = [v for v in views if v.id == view_id]
    if not view_:
        print(f"No coding game view found. Candidates: {[(v.data.title, v.id) for v in views]}")
        return
    view_ = view_[0]

    project_annotations = get_recent_annotations(po.id, 0)

    for_coding_game = []

    for task_res in project_annotations.annotations:
        annotations = task_res.annotations
        for annotation in annotations:
            for result in annotation.result:
                if result.from_name == "for_coding_game":
                    if result.value.choices[0] == "Yes":
                        p_id = task_res.data["platform_id"]
                        if p_id not in for_coding_game:
                            for_coding_game.append(p_id)

    build_view_with_filter_p_ids(SETTINGS.client, view_, for_coding_game)
    print("Coding game successfully updated")
    return po.id, view_id


@app.command()
def agreements(platform: Annotated[str, typer.Option()],
               language: Annotated[str, typer.Option()],
               accepted_ann_age: Annotated[
                   int, typer.Option(help="Download annotations if older than x hours")] = 2,
               min_num_coders: Annotated[int, typer.Option()] = 2
               ):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]

    conf = parse_label_config_xml(project_data["label_config"],
                                  project_id=project_id,
                                  include_text=True)

    annotations = get_recent_annotations(project_id, accepted_ann_age)

    data_extensions = None
    if (fi := SETTINGS.BASE_DATA_DIR / f"fixes/{project_id}.json").exists():
        data_extensions = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
    mp = MyProject(project_data=project_data,
                   annotation_structure=conf,
                   data_extensions=data_extensions,
                   raw_annotation_result=annotations)
    results = mp.calculate_results()
    mp.apply_extension(fillin_defaults=True)
    # print(results)

    check_col = ["any_harmful", "nature_text", "nature_visual", "val-expr_text", "val-expr_visual"]
    conflict = ["nature_text", "nature_visual", "val-expr_text", "val-expr_visual", "rel-value_text",
                "rel-value_visual",
                "nep_materiality_text", "nep_biological_text", "landscape-type_text",
                "basic-interaction_text",
                "nep_materiality_visual", "nep_biological_visual", "landscape-type_visual",
                "basic-interaction_visual"]
    all_rel = []
    # mp.annotation_structure.choices['nature_visual']
    for task in results.annotation_results:
        if task.num_coders < min_num_coders:
            continue
        res = task.data()
        vals = {c: (res.get(c) or [])[:min_num_coders] for c in check_col}

        for col in check_col:
            options = mp.annotation_structure.choices[col].options
            options = [c.alias if c.alias else c.value for c in options]
            default = data_extensions.fixes[col].default
            if default and default not in options:
                options.append(default)
            mp.annotation_structure.choices[col]
            ## todo: this would take only the first, for multiple choice
            vals[col] = [options.index(v[0]) for v in vals[col]]
        all_rel.append(vals)

    # print(all_rel)
    for col in check_col:
        print(col)
        col_data = [r[col] for r in all_rel]
        if all(not d for d in col_data):
            print(f"No data for : {col}")
            continue
        df = pd.DataFrame(col_data)
        # print(df)
        # if df.empty:
        #     print(f"No data for : {col}")
        #     continue
        cac_4raters = CAC(df)
        gwet_res = cac_4raters.gwet()
        # print(gwet_res)
        print(gwet_res["est"]["coefficient_value"])
    # print(results)


@app.command()
def generate_result_fixes_template(platform: Annotated[str, typer.Argument()],
                                   language: Annotated[str, typer.Argument()]):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]

    conf = parse_label_config_xml(project_data["label_config"],
                                  project_id=project_id,
                                  include_text=True)
    from ls_helper.funcs import generate_result_fixes_template as gen_fixes_template
    res_template = gen_fixes_template(project_id, conf)
    dest = Path(f"data/temp/result_fix_template_{platform}-{language}_{project_id}.json")
    dest.write_text(res_template.model_dump_json())
    print(f"file -> {dest.as_posix()}")


@app.command(short_help="Just needs to be run once, for each new LS project")
def setup_project_settings(platform: Annotated[str, typer.Option()],
                           language: Annotated[str, typer.Option()]):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]
    res = ls_client().patch_project(project_id, {
        "maximum_annotations": 2,
        "sampling": "Uniform sampling"
    })
    if res.status_code != 200:
        print("error updaing project settings")


if __name__ == "__main__":
    # clean ...ON VM
    # clean_project_task_files(33)
    # DONE
    # set_view_items("youtube", "en", "Old-Sentiment/Framing",
    #                Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/temp/yt_en_problematic_tasks.json"))

    # JUPP
    annotations_results("youtube", "en", 2)
    # CODING GAME
    # download_project_views("youtube", "en")
    # download_project_views("youtube","es")
    # update_coding_game("youtube", "es")
    agreements("youtube", "en")
    # generate_result_fixes_template("youtube","en")

    # setup_project_settings()
