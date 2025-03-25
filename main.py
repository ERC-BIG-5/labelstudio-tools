import json
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.agreements import calc_agreements, prepare_df
from ls_helper.ana_res import parse_label_config_xml
from ls_helper.annotation_timing import plot_date_distribution, annotation_total_over_time, \
    plot_cumulative_annotations, get_annotation_lead_times
from ls_helper.annotations import create_annotations_results, get_recent_annotations
from ls_helper.config_helper import check_config_update
from ls_helper.exp.build_configs import build_configs
from ls_helper.funcs import build_view_with_filter_p_ids
from ls_helper.models import ProjectOverview
from ls_helper.my_labelstudio_client.annot_master import prepare_df_for_agreement, calc_agreements2, prep_multi_select
from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import ProjectViewModel
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data

app = typer.Typer(name="Labelstudio helper")


def ls_client() -> LabelStudioBase:
    return LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)


def open_image_simple(image_path):
    # Convert to absolute path and URI format
    file_path = Path(image_path).absolute().as_uri()
    webbrowser.open(file_path)


def project_selection(platform: Optional[str] = None, language: Optional[str] = None) -> list[tuple[str, str, int]]:
    projects = ProjectOverview.projects()
    selection: list[tuple[str, str, int]] = []
    if not platform:
        for pl_name, pl in projects:
            # print(pl_name, pl)
            for lang, pl_lang in pl.items():
                # print(pl_lang)
                if language and lang != language:
                    continue
                if pl_lang.id:
                    selection.append((pl_name, lang, pl_lang.id))
    elif not language:
        pl = projects[platform]
        for lang, lang_pl in pl.items():
            if lang_pl.id:
                selection.append((platform, lang, lang_pl.id))
        # print(project_ids)
    else:
        project_id = ProjectOverview.projects().get_project_id(platform, language)
        selection = [(platform, language, project_id)]
    return selection


@app.command(short_help="[setup] Required for annotation result processing. needs project-data")
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


@app.command(short_help="[setup] Just needs to be run once, for each new LS project")
def setup_project_settings(platform: Annotated[str, typer.Option()],
                           language: Annotated[str, typer.Option()]):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]
    res = ls_client().patch_project(project_id, {
        "maximum_annotations": 2,
        "sampling": "Uniform sampling"
    })
    if not res:
        print("error updating project settings")


@app.command(
    short_help="[setup] run build_config function and copy it into 'labeling_configs_dir'. Run 'update_labeling_configs' afterward")
def generate_labeling_configs(
        platform: str, language: str):
    config_files = build_configs()
    check_config_update(config_files)
    pass  # TODO
    # platform_projects.
    # check_against_fixes(next_conf, )


@app.command(help="[ls maint] Upload labeling config")
def update_labeling_configs(
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None
):
    # todo, if we do that. save it
    # download_project_data(platform, language)

    client = ls_client()
    project_to_update = project_selection(platform, language)
    for idx, (platform, lang, id) in enumerate(project_to_update):
        label_config = (SETTINGS.labeling_configs_dir / f"{platform}.xml").read_text(encoding="utf-8")

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
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None
):
    projects_to_dl = project_selection(platform, language)

    for idx, (platform_, language_, project_id) in enumerate(projects_to_dl, start=1):
        project_data = ls_client().get_project(project_id)
        if not project_data:
            raise ValueError(f"No project found: {project_id}")
        else:
            _dir = SETTINGS.projects_dir / platform_
            _dir.mkdir(parents=True, exist_ok=True)
            dest = _dir / f"{language_}.json"
            dest.write_text(project_data.model_dump_json())
            print(f"{idx}/{len(projects_to_dl)} {dest}")


@app.command(short_help="[maint]")
def download_project_views(
        platform: Annotated[str, typer.Option()],
        language: Annotated[str, typer.Option()]) -> list[
    ProjectViewModel]:
    po = ProjectOverview.projects()
    project_id = po.get_project_id(platform, language)

    client = ls_client()
    views = client.get_project_views(project_id)
    dest = po.get_view_file(project_id)
    dest.write_text(json.dumps([v.model_dump() for v in views]))
    return views


@app.command(short_help="[plot] Plot the completed tasks over time")
def status(platform: Annotated[str, typer.Argument()],
           language: Annotated[str, typer.Argument()],
           name: Annotated[Optional[str], typer.Option()] = None,
           accepted_ann_age: Annotated[int, typer.Option(help="Download annotations if older than x hours")] = 6):
    from ls_helper import main_funcs
    main_funcs.status(platform, language, name, accepted_ann_age)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def total_over_time(platform: Annotated[str, typer.Argument()],
                    language: Annotated[str, typer.Argument()],
                    accepted_ann_age: Annotated[
                        int, typer.Option(help="Download annotations if older than x hours")] = 6):
    project_data = ProjectOverview.project_data(platform, language)
    project_id = project_data["id"]
    annotations = get_recent_annotations(project_id, accepted_ann_age)
    df = annotation_total_over_time(annotations)
    temp_file = plot_cumulative_annotations(df,
                                            f"{platform}/{language}: Cumulative Annotations Over Time")
    dest = SETTINGS.plots_dir / f"{platform}-{language}.png"
    shutil.copy(temp_file.name, dest)
    temp_file.close()
    open_image_simple(dest)


@app.command(short_help="[plot] Plot the total completed tasks over day")
def annotation_lead_times(project_id: Annotated[int, typer.Option()],
                          accepted_ann_age: Annotated[
                              int, typer.Option(help="Download annotations if older than x hours")] = 6):
    project_annotations = get_recent_annotations(project_id, accepted_ann_age)

    df = get_annotation_lead_times(project_annotations)
    temp_file = plot_date_distribution(df, y_col="lead_time")

    open_image_simple(temp_file.name)
    temp_file.close()


@app.command(short_help="[ls func]")
def set_view_items(platform: Annotated[str, typer.Option()],
                   language: Annotated[str, typer.Option()],
                   view_title: Annotated[str, typer.Option(help="search for view with this name")],
                   platform_ids_file: Annotated[Path, typer.Option()],
                   create_view: Annotated[Optional[bool], typer.Option()] = True
                   ):
    po = ProjectOverview.projects()
    views = po.get_views((platform, language))
    if not views and not create_view:
        print("No views found")
        return
    _view: ProjectViewModel = None
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
            ls_client().create_view()

    # check the file:
    if not platform_ids_file.exists():
        print(f"file not found: {platform_ids_file}")
        return
    platform_ids = json.load(platform_ids_file.open())
    assert isinstance(platform_ids, list)
    build_view_with_filter_p_ids(SETTINGS.client, _view, platform_ids)
    print("View successfully updated")


@app.command(short_help=f"[ls func]")
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


@app.command(short_help="[stats] Annotation basic results")
def annotations_results(platform: Annotated[str, typer.Argument()],
                        language: Annotated[str, typer.Argument()],
                        accepted_ann_age: Annotated[
                            int, typer.Option(help="Download annotations if older than x hours")] = 6,
                        min_coders: Annotated[int, typer.Option()] = 2) -> tuple[
    Path, str]:
    mp = create_annotations_results((platform, language), accepted_ann_age=accepted_ann_age)

    dest = SETTINGS.annotations_results_dir / f"{str(mp.project_id)}.csv"
    mp.results2csv(dest, with_defaults=True, min_coders=min_coders)
    print(f"annotation results -> {dest}")
    return dest, mp.raw_annotation_result.file_path.stem


@app.command(short_help="[stats] calculate general agreements stats")
def agreements(platform: Annotated[str, typer.Argument()],
               language: Annotated[str, typer.Argument()],
               accepted_ann_age: Annotated[
                   int, typer.Option(help="Download annotations if older than x hours")] = 2,
               min_num_coders: Annotated[int, typer.Option()] = 2
               ) -> dict[str, Path]:
    mp = create_annotations_results((platform, language), accepted_ann_age=accepted_ann_age)

    agreements_table_path, pid_data_file = calc_agreements(mp, min_num_coders)
    return {"agreements": agreements_table_path, "pids": pid_data_file}


if __name__ == "__main__":
    pass
    # crashes... FIX
    # agreements("youtube", "en")

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

    # TODO
    # generate_labeling_configs()

    # METHOD 3. straight to DF
    # get raw DF
    res = create_annotations_results(("twitter", "en"))
    #
    # deff = prepare_df_for_agreement(res,"framing")
    # calc_agreements2(deff)

    cats = ["nature_text", "val-expr_text", "val-expr_visual", "nep_materiality_text", "nep_biological_text",
            "landscape-type_text", "basic-interaction_text","media_relevant", "nep_materiality_visual_$", "nep_biological_visual_$", "landscape-type_visual_$",
            "basic-interaction_visual_$"]
    for cat in cats:
        # print(cat)
        type_ = res.annotation_structure.question_type(cat)
        if type_ == "single":
            deff = prepare_df_for_agreement(res, cat, False)
            fleiss, gwet = calc_agreements2(deff)
            print(f"{cat=} {fleiss=} {gwet=}")
        else:
            print(f"M {cat}")
            rdf = res.raw_annotation_df
            options = res.annotation_structure.choices.get(cat).raw_options_list()
            # print(options)
            # red_df = prep_multi_select(rdf, cat, options)
            pass

    exit()

    # METHOD 2.... TOO LONG
    res = create_annotations_results(("twitter", "en"))
    # res.results2csv(Path("t.csv"), with_defaults=True, min_coders=2)
    # # print(json.dumps(res.calc_annotation_result.model_dump(), indent=2))
    # # Path("d.json").write_text(res.model_dump_json(include={"annotation_results"},indent=2))
    # df = create_df(res)
    # df.to_csv(Path("df.csv"))
    #
    # multi_choice_options = {}
    #
    # for c_name, c in res.annotation_structure.choices.items():
    #     if c.choice == "multiple":
    #         multi_choice_options[c_name] = c.indices
    #
    pp_s, pp_m = prepare_df(df, True, {"nature_text", "nature_visual", "nep_materiality_text", "nep_biological_text",
                                       "landscape-type_text",
                                       "basic-interaction_text", "media_relevant"}, multi_choice_options)
    # df.to_csv(Path("df.csv"))
    #
    # cats = ["nature_text", "nature_visual",
    #         "nep_materiality_text", "nep_biological_text", "landscape-type_text", "basic-interaction_text",
    #         "media_relevant", "nep_materiality_visual_$", "nep_biological_visual_$", "landscape-type_visual_$",
    #         "basic-interaction_visual_$"]
    #
    # calc_agreements2(res, cats)
    # END METHOD 2
    # pp_s["nature_visual"].to_csv(Path("nature_visual.csv"))

    # p_df,  mapp_ = prepare_numeric_agreement(df)
    # print(p_df["nature_visual"].head(30))
    # p_df["nature_visual"].to_csv(Path("p_df.csv"))
    """
    single_df = df[df['question_type'] == 'single']

    # Pivot to get format for agreement calculation
    pivot_single = single_df.pivot_table(
        index=['task_id', 'item_id', 'question_id'],
        columns='coder_id',
        values='response'
    )
    """
