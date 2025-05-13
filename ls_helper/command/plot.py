from typing import Annotated, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS
from tools.project_logging import get_logger

logger = get_logger(__file__)

plot_app = typer.Typer(
    name="Plot",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


def basic_agreements(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        show: Annotated[bool, typer.Option()] = True
):
    po = get_project(id, alias)
    res = po.get_annotations_results()
    variables = {"nature_any", "nature_text", "nature_visual_any"}
    rename = {"nature_visual_any": "nature_visual"}
    _, data = res.clean_annotation_results(variables=variables)
    rows = []

    for p_id, res in data.items():
        if len(res) != 2:
            print(p_id, len(res))
            # todo make this a param!
            data[p_id] = res[:2]
        row = {rename.get(v, v): 0 for v in variables}
        for d in res:
            for v in variables:
                new_n = rename.get(v, v)
                row[new_n] += 1 if d.get(v, "No") == "Yes" else 0
        rows.append(row)

    df = pd.DataFrame(rows)

    def agreement_count_df(df_: pd.DataFrame, var: str) -> pd.DataFrame:
        df__ = df_[var].value_counts().to_frame().reset_index()
        df__[var] = df__[var].map({0: "agree_neg", 1: 'disagree', 2: "agree_pos"})
        return df__

    sns.set_theme(style="whitegrid")

    # figure 1
    df_nat_any = agreement_count_df(df, "nature_any")
    fig, ax = plt.subplots(figsize=(6, 15))
    sns.set_color_codes("pastel")
    sns.barplot(x="nature_any", y="count", data=df_nat_any,
                label="Total", color="b")

    sns.despine(left=True, bottom=True)
    if show:
        plt.show()

    fig.savefig(po.path_for(SETTINGS.plots_dir, f"nature_0_{po.id}", ext=".png"))

    # figure 2
    fig, ax = plt.subplots(figsize=(6, 15))
    df_nat_any_dis = df[df["nature_any"] == 1]
    df_nat_any_dis.drop(["nature_any"], axis=1, inplace=True)
    df_nat_any_dis["both"] = (df_nat_any_dis["nature_text"] & df_nat_any_dis["nature_visual"]).astype(int)
    df_nat_any_dis.loc[df_nat_any_dis['both'] == 1, 'nature_text'] = 0
    df_nat_any_dis.loc[df_nat_any_dis['both'] == 1, 'nature_visual'] = 0
    df_nat_any_dis_c = df_nat_any_dis.sum().to_frame().reset_index().rename(
        columns={"index": "nature", 0: "count"})
    sns.barplot(x="nature", y="count", data=df_nat_any_dis_c,
                label="Total", color="b")

    sns.despine(left=True, bottom=True)
    fig.savefig(po.path_for(SETTINGS.plots_dir, f"nature_1_{po.id}", ext=".png"))

    if show:
        plt.show()

    ###
    df_nat_any_ag = df[df["nature_any"] == 2].drop(["nature_any"], axis=1)
    df_nat_agg_p = df_nat_any_ag.groupby(["nature_text"]).value_counts().unstack()
    df_nat_agg_p.columns.rename("nature_visual", inplace=True)
    df_nat_agg_p.sort_index(axis=0, inplace=True, ascending=False)
    ##

    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_nat_agg_p, annot=True, linewidths=.5, ax=ax, cmap="crest")
    if show:
        plt.show()

    fig.savefig(po.path_for(SETTINGS.plots_dir, f"nature_2_{po.id}", ext=".png"))
