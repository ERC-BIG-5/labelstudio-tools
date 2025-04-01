import json
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.annotation_timing import plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.annotations import create_annotations_results, get_recent_annotations, reformat_for_datapipelines
from ls_helper.anot_master2 import analyze_coder_agreement
from ls_helper.config_helper import check_config_update
from ls_helper.exp.build_configs import build_configs
from ls_helper.funcs import build_view_with_filter_p_ids, build_platform_id_filter, get_variable_extensions
from ls_helper.models import MyProject
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import ProjectViewModel, ProjectViewCreate, ProjectViewDataModel
from ls_helper.new_models import platforms_overview2, get_p_access
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(name="Labelstudio helper", pretty_exceptions_show_locals=True)

"""
id_ = typer.Option(None, "--id"),
platform_ = typer.Option(None, "--platform", "-p"),
language_ = typer.Option(None, "--language", "-l"),
alias_ = typer.Option(None, "--alias", "-a"),
"""


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


@app.command(short_help="[setup] Required for annotation result processing. needs project-data")
def generate_result_fixes_template(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
):
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    project_id = po.id

    conf = po.get_structure()
    res_template = get_variable_extensions(conf)

    universal_fixes = read_data(SETTINGS.unifix_file_path)
    for k in res_template.fixes:
        if k in universal_fixes:
            # todo, can delete them?
            print(k)

    dest = Path(f"data/temp/result_fix_template_{project_id}.json")
    dest.write_text(res_template.model_dump_json())
    print(f"file -> {dest.as_posix()}")


@app.command(short_help="[setup] Just needs to be run once, for each new LS project")
def setup_project_settings(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None):
    po = platforms_overview2.get_project(get_p_access(id, platform, language, alias))
    values = ProjectMgmt.default_project_values()
    del values["color"]
    res = ls_client().patch_project(po.id, values)
    if not res:
        print("error updating project settings")


@app.command(
    short_help="[setup] run build_config function and copy it into 'labeling_configs_dir'. Run 'update_labeling_configs' afterward")
def generate_labeling_configs(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
):
    config_files = build_configs()
    check_config_update(config_files)
    pass  # TODO
    # platform_projects.
    # check_against_fixes(next_conf, )


@app.command(help="[ls maint] Upload labeling config")
def update_labeling_configs(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
):
    # todo, if we do that. save it
    # download_project_data(platform, language)
    client = ls_client()
    po = platforms_overview2.get_project(get_p_access(id, platform, language, alias))

    label_config = (SETTINGS.labeling_configs_dir / f"{po.platform}.xml").read_text(encoding="utf-8")

    resp = client.validate_project_labeling_config(id, label_config)
    if resp.status_code != 200:
        print(resp.status_code)
        print(resp.json())
        return
    res = client.patch_project(id, {"label_config": label_config})
    if not res:
        print(f"Could not update labeling config for {platform}/{language}/{id}")
        return
    print(f"updated labeling config for {platform}/{language}/{id}")


@app.command(short_help="[ls maint] Update tasks. Files must be matching lists of {id: , data:}")
# todo: more testing
def strict_update_project_tasks(new_data_file: Path,
                                existing_data_file: Optional[Path] = None):
    client = ls_client()
    new_data_list = json.loads(new_data_file.read_text(encoding="utf-8"))
    if existing_data_file:
        existing_data_list = json.loads(existing_data_file.read_text(encoding="utf-8"))
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


@app.command(
    short_help="[ls fixes] delete the json files from the local storage folder, from tasks that habe been deleted (not crucial)")
def clean_project_task_files(project_id: Annotated[int, typer.Option()],
                             title: Annotated[Optional[str], typer.Option()] = None,
                             just_check: Annotated[bool, typer.Option()] = False):
    # TODO, put this away. we rather just patch tasks per api
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


@app.command(short_help="[maint]")
def download_project_data(
        id: Annotated[int, typer.Option()] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        alias: Annotated[Optional[str], typer.Argument()] = None,
):
    p = platforms_overview2.get_project(get_p_access(id, platform, language, alias))

    project_data = ls_client().get_project(p.id)
    if not project_data:
        raise ValueError(f"No project found: {p.id}")
    else:
        dest = SETTINGS.projects_dir / f"{p.id}.json"
        dest.write_text(project_data.model_dump_json())


@app.command(short_help="[maint]")
def download_project_views(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
) -> list[
    ProjectViewModel]:
    p_a = get_p_access(id, alias, platform, language)
    po = platforms_overview2.get_project(p_a)

    client = ls_client()
    views = client.get_project_views(po.id)
    dest = po.get_view_file()
    dest.write_text(json.dumps([v.model_dump() for v in views]))
    return views


@app.command(short_help="[plot] Plot the completed tasks over time")
def status(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 6):
    from ls_helper import main_funcs
    main_funcs.status(get_p_access(id, alias, platform, language), accepted_ann_age)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def total_over_time(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")] = 6,
):
    print(get_p_access(id, alias, platform, language))
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    annotations = get_recent_annotations(po.id, accepted_ann_age)
    df = annotation_total_over_time(annotations)
    temp_file = plot_cumulative_annotations(df,
                                            f"{po.platform}/{po.language}: Cumulative Annotations Over Time")
    dest = SETTINGS.plots_dir / f"{platform}-{language}.png"
    shutil.copy(temp_file.name, dest)
    temp_file.close()
    open_image_simple(dest)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def annotation_lead_times(id: Annotated[int, typer.Option()] = None,
                          alias: Annotated[str, typer.Option("-a")] = None,
                          platform: Annotated[str, typer.Argument()] = None,
                          language: Annotated[str, typer.Argument()] = None,
                          accepted_ann_age: Annotated[
                              int, typer.Option(help="Download annotations if older than x hours")] = 6):
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    project_annotations = get_recent_annotations(po.id, accepted_ann_age)

    df = get_annotation_lead_times(project_annotations)
    temp_file = plot_date_distribution(df, y_col="lead_time")

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="[ls func]")
def set_view_items(
        view_title: Annotated[str, typer.Option(help="search for view with this name")],
        platform_ids_file: Annotated[Path, typer.Option()],
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        create_view: Annotated[Optional[bool], typer.Option()] = True
):
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    views = po.get_views()
    if not views and not create_view:
        print("No views found")
        return
    _view: Optional[ProjectViewModel] = None
    for view in views:
        if view.data.title == view_title:
            _view = view
            break
    if not _view:
        if not create_view:
            views_titles = [v.data.title for v in views]
            print(f"No views found: '{view_title}', candidates: {views_titles}")
            return
        else:  # create the view
            # todo, use utils func with id, title, adding in the defautl columns.
            ProjectMgmt.create_view(ProjectViewCreate(project=po.id, data=ProjectViewDataModel(title=view_title)))

    # check the file:
    if not platform_ids_file.exists():
        print(f"file not found: {platform_ids_file}")
        return
    platform_ids = json.load(platform_ids_file.open())
    assert isinstance(platform_ids, list)
    build_view_with_filter_p_ids(SETTINGS.client, _view, platform_ids)
    print("View successfully updated")


@app.command(short_help=f"[ls func]")
def update_coding_game(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
) -> Optional[tuple[int, int]]:
    """
    if successful sends back project_id, view_id

    """
    p_a = get_p_access(id, alias, platform, language)
    po = platforms_overview2.get_project(p_a)
    view_id = po.coding_game_view_id
    if not view_id:
        print("No views found for coding game")
        return None

    views = po.get_views()
    if not views:
        download_project_views(platform, language)
        views = po.get_views()
        # print("No views found for project. Call 'download_project_views' first")
        # return
    view_ = [v for v in views if v.id == view_id]
    if not view_:
        print(f"No coding game view found. Candidates: {[(v.data.title, v.id) for v in views]}")
        return None
    view_ = view_[0]

    project_annotations = get_recent_annotations(po.id, 0)

    for_coding_game = []
    # todo, rather get more processed object,
    for task_res in project_annotations:
        annotations = task_res.annotations
        for annotation in annotations:
            for result in annotation.result:
                # TODO: coding-game variable name, should be universal
                if result.from_name == "coding_game":
                    if result.value.choices[0] == "Yes":
                        p_id = task_res.data["platform_id"]
                        if p_id not in for_coding_game:
                            for_coding_game.append(p_id)

    build_view_with_filter_p_ids(SETTINGS.client, view_, for_coding_game)
    print("Coding game successfully updated")
    return po.id, view_id


@app.command(short_help="[stats] Annotation basic results")
def annotations_results(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")] = 6,
        min_coders: Annotated[int, typer.Option()] = 2) -> tuple[
    Path, str]:
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    po.check_fixes()
    mp = create_annotations_results(po, accepted_ann_age=accepted_ann_age)
    res = mp.flatten_annotation_results(min_coders)
    dest = SETTINGS.annotations_results_dir / f"{mp.project_id}.csv"
    res.to_csv(dest, index=False)
    print(f"annotation results -> {dest}")
    return dest


@app.command(short_help="[stats] calculate general agreements stats")
def agreements(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 2,
        min_num_coders: Annotated[int, typer.Option()] = 2
) -> tuple[Path, dict]:
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    mp = create_annotations_results(po, accepted_ann_age=accepted_ann_age)

    variables = {}

    fixes = mp.data_extensions.fixes
    for var, fix_info in fixes.items():
        if new_name := fixes[var].name_fix:
            name = new_name
        else:
            name = var

        default = fix_info.default
        type = mp.annotation_structure.question_type(name)
        if type in ["single", "multiple"]:
            options = mp.annotation_structure.choices[name].raw_options_list()
        else:
            options = []
        variables[name] = {
            "type": type,
            "options": options,
            "default": default
        }

    agreement_report = analyze_coder_agreement(mp.raw_annotation_df, mp.assignment_df, variables)
    dest: Path = (SETTINGS.agreements_dir / f"{mp.project_id}.json")
    print(f"agreement_report -> {dest.as_posix()}")
    dest.write_text(json.dumps(agreement_report))
    return dest, agreement_report



@app.command()
def add_project():
    plist = ls_client().projects_list()


@app.command()
def reformat_for_datapipelines(
        destination: Annotated[Path, typer.Argument()],
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
):
    """
    create a file with a dict platform_id: annotation, which can be ingested by the pipeline
    :param platform:
    :param language:
    :param destination:
    :return:
    """
    # does extra caluculation but ok.
    po = platforms_overview2.get_project(get_p_access(id, alias, platform, language))
    mp = create_annotations_results(po, True, 0)
    reformat_for_datapipelines(mp, destination)


@app.command()
def create_conflict_view(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Option()] = None,
        language: Annotated[str, typer.Option()] = None,
        variable: Annotated[str, typer.Option()] = None,
        variable_option: Annotated[str, typer.Option()] = None
):
    p_info = platforms_overview2.get_project(get_p_access(id, alias, platform, language))

    # TODO ALL THIS OUT: DONE IN CALCULATE-RESULTS
    # down from here till... should be somewhere...
    project_data = p_info.project_data

    conf = p_info.get_structure()
    data_extensions = p_info.data_extension
    mp = MyProject(
        platform=p_info.platform,
        language=p_info.language,
        project_data=project_data,
        annotation_structure=conf,
        data_extensions=data_extensions)

    mp.apply_extension()

    # just check existence
    _ = mp.annotation_structure.question_type(variable)

    agreement_data = json.loads((SETTINGS.agreements_dir / f"{p_info.id}.json").read_text())

    conflicts = agreement_data["conflicts"]
    relevant_conflicts_p_ids = [c["platform_id"] for c in conflicts if c["variable"] == variable]
    print(relevant_conflicts_p_ids)

    title = f"conflict:{variable}"
    ProjectMgmt.create_view(ProjectViewCreate.model_validate({"project": p_info.id, "data": {
        "title": title,
        "filters": build_platform_id_filter(relevant_conflicts_p_ids)}}))


@app.command()
def build_extension_index(
        take_all_defaults: Annotated[bool, typer.Option()] = True,
        project_ids: Annotated[list[int], typer.Option("-pid")] = None,
):
    from ls_helper.annot_extension import build_extension_index as _build_extension_index

    if project_ids:
        projects = [platforms_overview2.get_project(get_p_access(id)) for id in project_ids]
    elif take_all_defaults:
        projects = list(platforms_overview2.default_map.values())
    else:
        raise ValueError(f"Unclear parameter for build_extension_index")
    index = _build_extension_index(projects)
    dest = SETTINGS.temp_file_path / f"annot_ext_index_{'_'.join(str(p.id) for p in projects)}.json"
    dest.write_text(index.model_dump_json(indent=2))
    print(f"index saved to {dest}")


if __name__ == "__main__":
    build_extension_index(project_ids=[39, 43])
    exit()

    annotations_results(platform="twitter", language="en")
    agreements(platform="twitter", language="en")
    # download_project_views(alias="twitter-en-2")
    #
    # update_coding_game(alias="twitter-en-2")

    # download_project_data(id=41)
    # agre = agreements(alias="twitter-en-2")
    # create_conflict_view(alias="twitter-en-2", variable="nature_any")

    # down
    # generate_result_fixes_template(alias="twitter-en-2")
    exit()

    # reformat_for_datapipelines("twitter", "en",
    #                            Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/temp/twitter_en_res.json"))

    # pass
    # crashes... FIX
    # agreements("twitter", "en")
    # exit()
    # download_project_data("youtube", "en")

    # clean ...ON VM
    # clean_project_task_files(33)
    # DONE
    # set_view_items("youtube", "en", "Old-Sentiment/Framing",
    #                Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/temp/yt_en_problematic_tasks.json"))

    # JUPP
    # annotations_results("youtube", "en", 2)
    # annotations_results("twitter", "en", 2)
    # agreements("twitter", "en", 2)
    # CODING GAME
    # download_project_views("youtube", "en")
    # download_project_views("youtube", "en")
    # update_coding_game("youtube", "es")
    # agreements("youtube", "en")
    # generate_result_fixes_template("youtube","en")

    # setup_project_settings()

    #
    # download_project_data()
    # download_project_data("test")
    # generate_labeling_configs()
    # update_labeling_configs("test", "en")
