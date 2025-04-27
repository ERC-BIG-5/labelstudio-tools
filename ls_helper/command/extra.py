from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS
from tools.project_logging import get_logger

logger = get_logger(__file__)

extras_app = typer.Typer(
    name="Extra things", pretty_exceptions_show_locals=True
)


@extras_app.command(short_help="[stats] Annotation basic results")
def get_confusions(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
        min_coders: Annotated[int, typer.Option()] = 2,
) -> Path:
    po = get_project(id, alias, platform, language)
    po.validate_extensions()
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    df, _ = mp.get_annotation_df()
    # standard shape
    df = df[df["variable"].str.startswith("rel-value")].drop(["task_id", "ann_id", "ts", "type"], axis=1).reset_index(
        drop=True)
    # exploded shape
    df = df.set_index(['platform_id', 'user_id', "variable", "idx"])
    # df['rel-val-index'] = range(len(df))
    df = df.explode('value')

    df["value-index"] = df.groupby(level=['platform_id', 'user_id', "variable", "idx"]).cumcount()
    # sorted by re-value, and for each the confusion
    df = df.reset_index()

    # todo: separate conf out
    rel_values = ['personal-identity', 'cultural-identity', 'social-responsibility', 'social-cohesion', 'social-memory',
                  'social-relations', 'sense-of-place', 'sense-of-agency', 'spirituality', 'stewardship-principle',
                  'stewardship-eudaimonia', 'literacy', 'livelihoods', 'well-being', 'aesthetics', 'reciprocity',
                  'good-life', 'kinship']
    df['value_cat'] = pd.Categorical(df['value'], categories=rel_values, ordered=True)
    df = df.sort_values('value_cat').drop(columns=['value_cat'])
    pass
    # next, for each value, check, if there is a matching conf:
    # p_id,u_id, idx must be matching
    df.to_csv(po.path_for(SETTINGS.temp_file_path, alternative="rel-values_confusions", ext=".csv"))
