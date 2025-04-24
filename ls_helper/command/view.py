import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from ls_helper.funcs import build_view_with_filter_p_ids, download_project_views
from ls_helper.models.main_models import get_project, get_p_access
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import ProjectViewCreate, ProjectViewDataModel, ProjectViewModel
from ls_helper.project_mgmt import ProjectMgmt
from ls_helper.settings import SETTINGS
from tools.project_logging import get_logger

logger = get_logger(__file__)

view_app = typer.Typer(name="Project View related commands", pretty_exceptions_show_locals=False)



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



@view_app.command()
def delete_view(view_id: int):
    ls_client().delete_view(view_id)

