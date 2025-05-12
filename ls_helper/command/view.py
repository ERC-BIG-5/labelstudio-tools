import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from deprecated import deprecated

from ls_helper.funcs import (
    build_view_with_filter_p_ids,
    build_platform_id_filter,
)
from ls_helper.models.main_models import get_project
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectViewCreate,
    ProjectViewDataModel,
    ProjectViewModel,
)
from ls_helper.settings import SETTINGS
from tools.files import read_data
from tools.project_logging import get_logger

logger = get_logger(__file__)

view_app = typer.Typer(
    name="Project View related commands", pretty_exceptions_show_locals=False
)


@view_app.command(short_help="[ls func]")
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
    po = get_project(id, alias, platform, language)
    logger.info(po.alias)
    view_id = po.coding_game_view_id
    if not view_id:
        print("No views found for coding game")
        view = po.create_view(
            ProjectViewCreate.model_validate(
                {"project": po.id, "data": {"title": "Coding Game"}}
            )
        )
        view_id = view.id

    if refresh_views:
        po.refresh_views()
    views = po.get_views()
    if not views:
        download_project_views(id, alias, platform, language)
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

    # project_annotations = _get_recent_annotations(po.id, accepted_ann_age)
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    # project_annotations = _get_recent_annotations(po.id, 0)

    ann = mp.raw_annotation_df.copy()
    ann = ann[ann["variable"] == "coding-game"]
    ann = mp.simplify_single_choices(ann)
    platform_ids = ann[ann["single_value"] == "Yes"]["platform_id"].tolist()
    build_view_with_filter_p_ids(ls_client(), view_, platform_ids)
    logger.info(f"Set {len(platform_ids)} to the coding game of {po.alias}")
    return po.id, view_id


@view_app.command(short_help="[ls func]")
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
            po.create_view(
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


@view_app.command()
def delete_view(view_id: int):
    ls_client().delete_view(view_id)


@view_app.command(short_help="Download the views of a project")
def download_project_views(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
) -> list[ProjectViewModel]:
    po = get_project(id, alias, platform, language)
    views = po.refresh_views()
    logger.debug(f"view file -> {po.path_for(SETTINGS.view_dir)}")
    return views


@deprecated(reason="we can use annotation.add_conflicts_to_tasks instead")
@view_app.command(short_help="create or update a view for variable conflict")
def create_conflict_view(
        variable: Annotated[str, typer.Option()],
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
        variable_option: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    conflicts = read_data(
        po.path_for(SETTINGS.agreements_dir, alternative=f"{po.id}_conflicts")
    )
    if variable not in conflicts:
        print(f"No conflict data for {variable}. Wrong variable-name?")
    var_conflicts = conflicts[variable]
    conflict_task_ids = []
    if variable_option:
        if variable_option not in var_conflicts:
            print(
                f"No conflict data for {variable}. Wrong option. Options are: {list(var_conflicts.keys())}"
            )
        conflict_task_ids = var_conflicts[variable_option]
    else:
        for option in var_conflicts.values():
            conflict_task_ids.extend(option["conflict"])

    # we have to limit it...
    conflict_task_ids = conflict_task_ids[:30]

    title = f"conflict:{variable}"
    view = po.create_view(
        ProjectViewCreate.model_validate(
            {
                "project": po.id,
                "data": {
                    "title": title,
                    "filters": build_platform_id_filter(
                        conflict_task_ids, "task_id"
                    ),
                },
            }
        )
    )
    url = f"{SETTINGS.LS_HOSTNAME}/projects/{po.id}/data?tab={view.id}"
    print(url)
    return url
