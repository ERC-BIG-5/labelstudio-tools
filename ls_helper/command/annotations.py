from pathlib import Path
from typing import Annotated, Optional

import typer
from tools.project_logging import get_logger

from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS

logger = get_logger(__file__)

annotations_app = typer.Typer(
    name="Annotations", pretty_exceptions_show_locals=True
)


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
