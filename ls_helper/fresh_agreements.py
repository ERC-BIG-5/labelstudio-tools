import re
from typing import Optional

import irrCAC.raw
import pandas as pd
from numpy import nan

from ls_helper.new_models import ProjectData, get_project

from pandas import DataFrame


def find_groups(variables):
    pattern = re.compile(r'^(.+)_(\d+)$')

    groups: dict[str, dict[str, int]] = {}
    for var in variables:
        match = re.match(pattern, var)
        if match:
            base_name, idx = match.groups()
            idx = int(idx)
            # print(var, base_name,idx)
            groups.setdefault(base_name, {})[var] = idx

    return groups


def prepare_indexed_var(df, df_in) -> DataFrame:
    return df.merge(df_in, on="variable", how='left')


def select_variables(df, variables: list[str], groups: dict[str, dict[str, int]]) -> dict[str, DataFrame]:
    group_names = list(groups.keys())
    selected_orig_vars = []
    has_groups = False
    assignments = {}

    # collect all originals
    # todo: put that somewhere else?
    # iter through all vars
    for var in variables:
        # its a group var!
        if var in group_names:
            # number of group items
            idx_vars = list(groups[var].keys())
            # add the items to keep
            selected_orig_vars.extend(idx_vars)
            # add orig name with
            # todo: make this a dict not list/tuple
            assignments.update(
                {idx_v: [var, groups[var][idx_v]] for idx_v in idx_vars}
            )
            has_groups = True
            # print(assignments)
        else:
            selected_orig_vars.append(var)

    # select all relevant variables
    orig_vars_groups = df[df["variable"].isin(selected_orig_vars)].groupby("variable")

    if not has_groups:
        return {name: group for name, group in orig_vars_groups}

    result_groups: dict[str, DataFrame] = {}
    for variable, group_df in orig_vars_groups:
        if variable in assignments:
            groupe_name = assignments[variable][0]
            group_df["variable"] = groupe_name  # group-name
            group_df["idx"] = assignments[variable][1]  # index
            if groupe_name not in result_groups:
                # print(f"adding {gn=}")
                result_groups[groupe_name] = group_df
            else:
                result_groups[groupe_name] = pd.concat([result_groups[groupe_name], group_df])
        else:
            result_groups[variable] = group_df

    for rg in result_groups.values():
        rg["idx"] = rg["idx"].astype(int)
    return result_groups


def prepare_single_select(df):
    return df.explode("value")

def calc_agreement(df) -> float:
    if len(df) == 0:
        return nan
    df = df.reset_index()
    index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
    pv_df = df.pivot(columns="user_id", values="value", index=index)
    try:
        cac = irrCAC.raw.CAC(pv_df)
        return cac.gwet()["est"]["coefficient_value"]
    except (ValueError, ZeroDivisionError):
        return nan


def agreement_calc(po: ProjectData, variables: list[str],
                   force_default: Optional[str] = "NONE", max_coders: int = 2):
    df = po.get_annotations_results().raw_annotation_df.copy()
    #df_ts = df[["task_id", "user_id", "ts"]]
    #df_ts.set_index(["task_id", "user_id"], inplace=True)
    #df_ts = df_ts[~df_ts.index.duplicated(keep='first')]

    df = df.rename(columns={"category": "variable"})
    df = df.drop(["ann_id", "platform_id"], axis=1)

    df = df.set_index(["task_id", "user_id"])

    groups = find_groups(list(po.variable_extensions.extensions.keys()))
    variables_dfs = select_variables(df, variables, groups)
    for n, df in variables_dfs.items():
        print(n, len(df))

    def prepare_var(df, force_default: Optional[str] = None):
        if "variable" in df.columns:
            df = df.drop("variable", axis=1)
        df = df.groupby("task_id").filter(lambda x: len(x.index.get_level_values(1).unique()) > 1)

        # this part to throw out all with more than x coder. get the times, and only keep the latest 2.
        #df = df.join(df_ts)
        #df = df.sort_values("ts", ascending=False).groupby(level=0).head(max_coders).sort_index()
        if df.iloc[0]["type"] == "single":
            df = prepare_single_select(df)
        else:
            pass  # todo
        #df = df.drop("type", axis=1)
        df.fillna(force_default, inplace=True)
        return df

    def time_move(df):
        if "ts" not in df.columns:
            return df

        df['date'] = df['ts'].dt.date
        return df
        # Plot
        """
        df = df.sort_values('ts')
        daily_counts = df.groupby('date').size().cumsum()
        di = daily_counts.plot(
            figsize=(10, 6),
            title='Cumulative Number of Rows Over Time'
        )
        plt.show()
        """

    for var, v_df in variables_dfs.items():
        v_df = prepare_var(v_df, force_default).reset_index()

        calc_agreement(v_df)

        v_df = time_move(v_df)
        for day in sorted(v_df.date.unique()):
            day_v_df = v_df[v_df["date"] <= day]

            day_v_df = day_v_df.set_index(["task_id", "user_id"]).groupby("task_id").filter(lambda x: len(x.index.get_level_values(1).unique()) > 1)
            print(day, len(day_v_df), calc_agreement(day_v_df))
        # minimal...

        index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
        min_index = index + ["user_id"]
        df_min = v_df.copy().set_index(min_index)
        df_min['position'] = df_min.groupby(level=0).cumcount()
        df_min = df_min.droplevel('user_id', axis=0)



if __name__ == "__main__":
    po = get_project(43)
    agreement_calc(po, ["nature_any", "nature_text","nature_visual", "nep_material_visual"])
