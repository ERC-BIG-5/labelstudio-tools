import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.command.annotations import annotations_app, agreements
from ls_helper.command.backup import backup_app
from ls_helper.command.extra import extras_app
from ls_helper.command.labeling_conf import labeling_conf_app
from ls_helper.command.pipeline import pipeline_app
from ls_helper.command.project_setup import project_app
from ls_helper.command.setup import (
    setup_app,
)
from ls_helper.command.task import task_add_predictions, task_app
from ls_helper.command.view import view_app
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.models.interface_models import IChoices, InterfaceData
from ls_helper.models.main_models import (
    get_project,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.files import save_yaml
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(
    name="Labelstudio helper", pretty_exceptions_show_locals=False
)

app.add_typer(
    setup_app,
    name="setup",
    short_help="Commands related to initializing the project",
)
app.add_typer(
    project_app,
    name="project",
    short_help="Commands related to project setup and maintenance",
)
app.add_typer(
    backup_app,
    name="backup",
    short_help="Commands related to bulkbacking up projects and annotations",
)
app.add_typer(
    labeling_conf_app,
    name="labeling-conf",
    short_help="Commands related to building, validating and uploading project label configurations",
)
app.add_typer(
    task_app,
    name="task",
    short_help="Commands related to downloading, creating and patching project tasks",
)
app.add_typer(
    view_app, name="view", short_help="Commands related to project views"
)
app.add_typer(
    annotations_app,
    name="annotations",
    short_help="Commands related to downloading and analyzing annotations",
)
app.add_typer(
    pipeline_app,
    name="pipeline",
    short_help="Commands related to interaction with the Pipeline package",
)
app.add_typer(
    extras_app,
    name="extras",
    short_help="Some extra commands: [relational-values confusions]",
)


@app.command(
    name="strict task update",
    short_help="[ls maint] Update tasks. Files must be matching lists of {id: , data:}",
)
# todo: more testing
def strict_update_project_tasks(
    new_data_file: Path, existing_data_file: Optional[Path] = None
):
    raise NotImplementedError("client.patch_task parameters changed")
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
        ls_client().patch_task(t["id"], t["data"])

    print(f"{len(new_data_list)} tasks updated")


<<<<<<< HEAD
@deprecated
@app.command(
    short_help="[ls fixes] delete the json files from the local storage folder, from tasks that habe been deleted (not crucial)"
)
def clean_project_task_files(
    project_id: Annotated[int, typer.Option()],
    title: Annotated[Optional[str], typer.Option()] = None,
    just_check: Annotated[bool, typer.Option()] = False,
):
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
    id: Annotated[Optional[int], typer.Option(help="project id")] = None,
    alias: Annotated[Optional[str], typer.Argument()] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
) -> ProjectModel:
    po = get_project(id, alias, platform, language)
    project_data = ls_client().get_project(po.id)

    if not project_data:
        raise ValueError(f"No project found: {po.id}")
    po.save_project_data(project_data)
    return project_data


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


@app.command(short_help="[ls func]")
def set_view_items(
    view_title: Annotated[
        str, typer.Option(help="search for view with this name")
    ],
    platform_ids_file: Annotated[Path, typer.Option()],
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    create_view: Annotated[Optional[bool], typer.Option()] = True,
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
            print(
                f"No views found: '{view_title}', candidates: {views_titles}"
            )
            return
        else:  # create the view
            # todo, use utils func with id, title, adding in the defautl columns.
            ProjectMgmt.create_view(
                ProjectViewCreate(
                    project=po.id, data=ProjectViewDataModel(title=view_title)
                )
            )

    # check the file:
    if not platform_ids_file.exists():
        print(f"file not found: {platform_ids_file}")
        return
    platform_ids = json.load(platform_ids_file.open())
    assert isinstance(platform_ids, list)
    build_view_with_filter_p_ids(SETTINGS.client, _view, platform_ids)
    print("View successfully updated")


@app.command(short_help="[ls func]")
def update_coding_game(
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
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
        print(
            f"No coding game view found. Candidates: {[(v.data.title, v.id) for v in views]}"
        )
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
    id: Annotated[Optional[int], typer.Option()] = None,
    alias: Annotated[Optional[str], typer.Option("-a")] = None,
    platform: Annotated[Optional[str], typer.Argument()] = None,
    language: Annotated[Optional[str], typer.Argument()] = None,
    accepted_ann_age: Annotated[
        int, typer.Option(help="Download annotations if older than x hours")
    ] = 6,
    max_num_coders: Annotated[int, typer.Option()] = 2,
    variables: Annotated[Optional[list[str]], typer.Argument()] = None,
) -> Path:
    dest, agreement_report = (
        get_project(id, alias, platform, language)
        .get_annotations_results(accepted_ann_age=accepted_ann_age)
        .get_coder_agreements(max_num_coders, variables, True)
    )

    return dest


@app.command()
def create_project(
    title: Annotated[str, typer.Option()],
    alias: Annotated[str, typer.Option()],
    platform: Annotated[str, typer.Option()],
    language: Annotated[str, typer.Option()],
):
    platforms_overview.create(
        ProjectCreate(
            title=title,
            platform=platform,
            language=language,
            alias=alias,
            default=False,
        )
    )


=======
>>>>>>> 881e5f891475142ff5331315bf2408f9d28677a8
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
    from_built: Annotated[
        bool,
        typer.Option(False, help="Use the built instead of the project-data"),
    ] = False,
):
    po = get_project(id, alias, platform, language)

    if from_built:
        config: InterfaceData = parse_label_config_xml(
            po.path_for(
                SETTINGS.built_labeling_configs, ext=".xml"
            ).read_text()
        )
    else:
        config = po.raw_interface_struct

    interface_data = [
        {
            "name": k,
            "required": v.required,
            "choice_type": str(v.choice)
            if isinstance(v, IChoices)
            else "text",
            "choices": v.raw_options_list() if isinstance(v, IChoices) else "",
        }
        for k, v in config.ordered_fields_map.items()
    ]

    # yaml.dump(interface_data, po.path_for(SETTINGS.temp_file_path, ext=".yaml").open("w"), indent=True, default_flow_style=False, encoding="utf-8")
    save_yaml(
        po.path_for(SETTINGS.temp_file_path, ext=".yaml"), interface_data
    )
    return interface_data


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


@app.command(name="overview", short_help="Overview of all commands")
def overview():
    def _print_commands(current_app, prefix="", indent=0):
        """Recursively print commands and subcommands."""
        indent_str = "  " * indent

        # Print commands at this level
        for cmd in current_app.registered_commands:
            name = cmd.name or cmd.callback.__name__
            help_text = cmd.short_help or ""
            print(f"{indent_str}• '{name}' - {help_text}")

        # Find and process all subapps
        for group in current_app.registered_groups:
            subapp_name = group.name
            subapp = group.typer_instance
            print(f"{indent_str}▼ {prefix}{subapp_name}")
            _print_commands(subapp, f"{prefix}{subapp_name} ", indent + 1)

    # Start the recursive printing from the main app
    _print_commands(app)


if __name__ == "__main__":
    twitter = "twitter"
    youtube = "youtube"
    en = "en"
    es = "es"
    tw_es = {"platform": twitter, "language": es}
    yt_en4 = {"id": 50}
    _default = tw_es

    # setup

    # setup.add_projects()

    # this will work, since there is just one spanish twitter (so it's set to default)

    if False:
        agreements(
            **{"alias": "twitter-es-4"},
            accepted_ann_age=200,
            # variables=["coding-game"],
            exclude_variables=[
                "rel-value_text_conf_aesthetics",
                "rel-value_text_conf_cultural-identity",
                "rel-value_text_conf_social-cohesion",
                "rel-value_text_conf_good-life",
                "rel-value_text_conf_kinship",
                "rel-value_text_conf_livelihoods",
                "rel-value_text_conf_personal-identity",
                "rel-value_text_conf_reciprocity",
                "rel-value_text_conf_sense-of-agency",
                "rel-value_text_conf_sense-of-place",
                "rel-value_text_conf_social-relations",
                "rel-value_text_conf_social-responsibility",
                "rel-value_text_conf_spirituality",
                "rel-value_text_conf_stewardship-principle",
                "rel-value_text_conf_well-being",
                "rel-value_visual_conf_aesthetics",
                "rel-value_visual_conf_cultural-identity",
                "rel-value_visual_conf_social-cohesion",
                "rel-value_visual_conf_good-life",
                "rel-value_visual_conf_kinship",
                "rel-value_visual_conf_livelihoods",
                "rel-value_visual_conf_personal-identity",
                "rel-value_visual_conf_reciprocity",
                "rel-value_visual_conf_sense-of-agency",
                "rel-value_visual_conf_sense-of-place",
                "rel-value_visual_conf_social-relations",
                "rel-value_visual_conf_social-responsibility",
                "rel-value_visual_conf_spirituality",
                "rel-value_visual_conf_stewardship-principle",
                "rel-value_visual_conf_well-being",
            ],
        )

    # project_setup.generate_variable_extensions_template(id=50)
    # project_setup.generate_variable_extensions_template(id=51)

    # labeling_conf.build_ls_labeling_interface(id=53)

    # print(get_variables_info(53, from_built=True))

    # labeling_conf.build_extension_index(False,[51,50])
    # labeling_conf.build_ls_labeling_interface(53)
    # labeling_conf.update_labeling_config(53)
    """project_setup.create_project(
        title="Twitter - ES - protocol.v5",
        alias="twitter-es-5",
        platform="twitter",
        language="en",
    )"""
    # project_setup.generate_variable_extensions_template(53)
    # project_setup.generate_variable_extensions_template(54)

    # create_conflict_view("nature_any",**{"alias": "twitter-es-4"})
    # get_tasks(**{"alias": "twitter-es-4"})
    # add_conflicts_to_tasks(**{"alias": "twitter-es-4"})

    """
    # for creating/testing version 5 of the protocol
    from ls_helper.command import labeling_conf

    p, valid = labeling_conf.build_ls_labeling_interface(**{"id": 53})

    if valid:
        labeling_conf.update_labeling_config(**{"id": 53})

    yaml.dump(get_variables_info(id=53), Path("53.yaml").open("w", encoding="utf-8"))
    """
    # get_confusions(id=51)
    # add_conflicts_to_tasks(id=51)
    # get_confusions(id=51)
    # update_coding_game(id=51)
    # pipeline.reformat_for_datapipelines(alias="twitter-es-4", accepted_ann_age=300)

    overview()
