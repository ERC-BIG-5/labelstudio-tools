import json
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer
from deepdiff import DeepDiff
from tqdm import tqdm

from ls_helper.agreements import fix_users, AgreementReport, SingleChoiceAgreement
from ls_helper.annotation_timing import plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.command import annotations, labeling_conf
from ls_helper.command.labeling_conf import labeling_conf_app
from ls_helper.command.pipeline import pipeline_app
from ls_helper.command.setup import generate_variable_extensions_template
from ls_helper.config_helper import check_config_update, parse_label_config_xml
from ls_helper.exp.build_configs import build_configs
from ls_helper.funcs import build_view_with_filter_p_ids, build_platform_id_filter
from ls_helper.models.interface_models import InterfaceData, ProjectVariableExtensions, FieldExtension
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import ProjectViewModel, ProjectViewCreate, ProjectViewDataModel
from ls_helper.new_models import platforms_overview, get_p_access, ProjectCreate, get_project
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(name="Labelstudio helper", pretty_exceptions_show_locals=True)

app.add_typer(labeling_conf_app, name="label_conf")
app.add_typer(pipeline_app, name="pipeline")
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


@app.command(short_help="[setup] Just needs to be run once, for each new LS project")
def setup_project_settings(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None):
    po = get_project(id, alias, platform, language)
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
    existing_task_files = [f.title for f in existing_task_files]
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
        alias: Annotated[Optional[str], typer.Argument()] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
):
    po = get_project(id, alias, platform, language)
    project_data = ls_client().get_project(po.id)

    if not project_data:
        raise ValueError(f"No project found: {po.id}")
    po.save_project_data(project_data)



@app.command(short_help="[maint]")
def download_project_views(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None
) -> list[
    ProjectViewModel]:
    p_a = get_p_access(id, alias, platform, language)
    po = get_project(p_a)
    views = ProjectMgmt.refresh_views(po)
    logger.debug(f"view file -> {po.view_file}")
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

    po = get_project(id, alias, platform, language)
    po.validate_extensions()
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    # todo, this is not nice lookin ... lol
    res = mp.basic_flatten_results(1)
    # just for checking...
    client = ls_client()
    users = client.get_users()
    fix_users(res, {u.id: u.username for u in users})
    print(res["user_id"].value_counts())


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
    po = get_project(id, alias, platform, language)
    annotations = ProjectMgmt.get_recent_annotations(po.id, accepted_ann_age)
    df = annotation_total_over_time(annotations)
    temp_file = plot_cumulative_annotations(df,
                                            f"{po.platform}/{po.language}: Cumulative Annotations Over Time")
    dest = SETTINGS.plots_dir / f"{platform}-{language}.png"
    shutil.copy(temp_file.title, dest)
    temp_file.close()
    open_image_simple(dest)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def annotation_lead_times(id: Annotated[int, typer.Option()] = None,
                          alias: Annotated[str, typer.Option("-a")] = None,
                          platform: Annotated[str, typer.Argument()] = None,
                          language: Annotated[str, typer.Argument()] = None,
                          accepted_ann_age: Annotated[
                              int, typer.Option(help="Download annotations if older than x hours")] = 6):
    po = get_project(id, alias, platform, language)
    project_annotations = ProjectMgmt.get_recent_annotations(po.id, accepted_ann_age)

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
    po = get_project(id, alias, platform, language)
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
        accepted_ann_age: Annotated[int, typer.Option("-age")] = 6,
        refresh_views: Annotated[bool, typer.Option("-r")] = False,
) -> Optional[tuple[int, int]]:
    """
    if successful sends back project_id, view_id

    """
    p_a = get_p_access(id, alias, platform, language)
    po = get_project(p_a)
    logger.info(po.alias)
    view_id = po.coding_game_view_id
    if not view_id:
        print("No views found for coding game")
        return None

    if refresh_views:
        ProjectMgmt.refresh_views(po)
    views = po.get_views()
    if not views:
        download_project_views(platform, language)
        views = po.get_views()
        # print("No views found for project. Call 'download_project_views' first")
        # return
    view_ = [v for v in views if v.id == view_id]
    if not view_:
        # todo: create view
        print(f"No coding game view found. Candidates: {[(v.data.title, v.id) for v in views]}")
        return None
    view_ = view_[0]

    po = get_project(id, alias, platform, language)
    # project_annotations = _get_recent_annotations(po.id, accepted_ann_age)
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    # project_annotations = _get_recent_annotations(po.id, 0)

    ann = mp.raw_annotation_df.copy()
    ann = ann[ann["category"] == "coding-game"]
    ann = mp.simplify_single_choices(ann)
    platform_ids = ann[ann["single_value"] == "Yes"]["platform_id"].tolist()
    build_view_with_filter_p_ids(SETTINGS.client, view_, platform_ids)
    logger.info(f"Set {len(platform_ids)} to the coding game of {po.alias}")
    return po.id, view_id


@app.command(short_help="[stats] calculate general agreements stats")
def agreements(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Argument()] = None,
        language: Annotated[str, typer.Argument()] = None,
        accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 6,
        min_num_coders: Annotated[int, typer.Option()] = 2,
        variables: Annotated[list[str], typer.Argument()] = None

) -> tuple[Path, AgreementReport]:
    dest, agreement_report = (get_project(id, alias, platform, language)
                              .get_annotations_results(
        accepted_ann_age=accepted_ann_age)
                              .get_coder_agreements(min_num_coders, variables, True))
    return dest, agreement_report


@app.command()
def create_project(
        title: Annotated[str, typer.Option()],
        platform: Annotated[str, typer.Option()],
        language: Annotated[str, typer.Option()],
        alias: Annotated[str, typer.Option()] = None):
    platforms_overview.add_project(ProjectCreate(
        title=title,
        platform=platform,
        language=language,
        alias=alias,
        default=False
    ))


def get_all_variable_names(
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Option()] = None,
        language: Annotated[str, typer.Option()] = None
):
    po = get_project(id, alias, platform, language)
    # todo redo and test...
    struct = po.raw_interface_struct(include_text=False, apply_extension=True)
    return list(struct.orig_choices.keys()) + struct.free_text


@app.command()
def create_conflict_view(
        variable: Annotated[str, typer.Option()],
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Option()] = None,
        language: Annotated[str, typer.Option()] = None,
        variable_option: Annotated[str, typer.Option()] = None
):
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


@app.command()
def build_extension_index(
        take_all_defaults: Annotated[bool, typer.Option(help="take default projects (pl/lang)")] = True,
        project_ids: Annotated[list[int], typer.Option("-pid")] = None,
):
    """
    Checks
    :param take_all_defaults:
    :param project_ids:
    :return:
    """
    from ls_helper.annot_extension import build_extension_index as _build_extension_index

    if project_ids:
        projects = [get_project(id) for id in project_ids]
    elif take_all_defaults:
        projects = list(platforms_overview.default_map.values())
    else:
        raise ValueError(f"Unclear parameter for build_extension_index")
    index = _build_extension_index(projects)
    dest = SETTINGS.temp_file_path / f"annot_ext_index_{'_'.join(str(p.id) for p in projects)}.json"
    dest.write_text(index.model_dump_json(indent=2))
    print(f"index saved to {dest}")


@app.command()
def delete_view(view_id: int):
    ls_client().delete_view(view_id)


@app.command()
def check_labelling_config(
        build_file_name: str,
        id: Annotated[int, typer.Option()] = None,
        alias: Annotated[str, typer.Option("-a")] = None,
        platform: Annotated[str, typer.Option()] = None,
        language: Annotated[str, typer.Option()] = None
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


if __name__ == "__main__":
    twitter = "twitter"
    youtube = "youtube"
    en = "en"
    tw_en = {"platform": twitter, "language": en}
    yt_en4 = {"id": 50}
    _default = tw_en

    # generate_result_fixes_template(**_default)
    # build_ls_labeling_interface(Path("twitter-2.json"))
    # build_extension_index(project_ids=[39, 43])
    # exit()
    # status(**_default)
    # annotations(**_default)
    # download_project_data(**_default)
    #agreements(**_default, accepted_ann_age=0)
    # delete_view(110)
    # download_project_views(**_default)
    # create_conflict_view(**_default, variable="landscape-type_text")
    # create_conflict_view(**_default, variable="landscape-type_visual")
    # create_conflict_view("nature_text",**_default)
    # update_coding_game(**_default)

    # alternative builts possible
    # sub apps:
    #labeling_conf
    # pipeline
    # task
    # annotations
    #generate_variable_extensions_template(50)
    #build_extension_index(project_ids=[50,43,33,39])
    #annotations.annotations(platform="twitter", language="en")

    # setup_project_settings(platform="youtube", language="en")
    #
    labeling_conf.build_ls_labeling_interface(platform="youtube", language="en")
    labeling_conf.update_labeling_config(platform="youtube", language="en")
    # pipeline.reformat_for_datapipelines(33,0)
    # check_labelling_config("twitter_reduced", **_default)

    # DONE!!
    # task.create_tasks(Path("/home/rsoleyma/projects/DataPipeline/data/ls_tasks/youtube-en-4"),
    #                   True,
    #                   None, "youtube-en-4")
