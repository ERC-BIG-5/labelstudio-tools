import re
from datetime import date
from typing import Optional, Annotated, Any, Literal, Generator

import irrCAC.raw
import pandas as pd
from deprecated.classic import deprecated
from numpy import nan
from pandas import DataFrame
from pydantic import BaseModel

from ls_helper.new_models import ProjectData, get_project


# experimental...
class DFAgreementsInitModel(BaseModel):
    task_id: int
    variable: Annotated[int, {"dtype": "category"}]


class Agreements():

    def __init__(self, po: ProjectData):
        self.po = po
        # group-name: variable-name: index
        self._groups: dict[str, dict[str,int]] = self.find_groups(list(po.variable_extensions.extensions.keys()))
        self._init_df: Optional[DataFrame] = self.create_init_df()


    @staticmethod
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

    @deprecated
    def prepare_indexed_var(self, df, df_in) -> DataFrame:
        return df.merge(df_in, on="variable", how='left')

    @staticmethod
    def base_df(df: DataFrame) -> DataFrame:
        index_ = df.index.names
        if "task_id" in index_ and "user_id" in index_:
            return df
        else:
            return df.set_index(["task_id", "user_id"])


    def select_variables(self,
                         variables: list[str],
                         always_as_dict: bool = True) -> dict[str, DataFrame] | DataFrame:
        df = self.get_init_df()
        group_names = list(self._groups.keys())
        selected_orig_vars = []
        has_groups = False
        assignments = {}

        # collect all originals
        # todo: put that somewhere else? Another model
        # iter through all vars
        for var in variables:
            # its a group var!
            if var in group_names:
                # number of group items
                idx_vars = list(self._groups[var].keys())
                # add the items to keep
                selected_orig_vars.extend(idx_vars)
                # add orig name with
                assignments.update(
                    {idx_v: {"group": var, "gr_variable": self._groups[var][idx_v]} for idx_v in idx_vars}
                )
                has_groups = True
                # print(assignments)
            else:
                selected_orig_vars.append(var)

        # select all relevant variables
        orig_vars_groups = df[df["variable"].isin(selected_orig_vars)].groupby("variable")

        if not has_groups:
            group_dict = {name: group for name, group in orig_vars_groups}
            if not always_as_dict and len(group_dict) == 1:
                return list(group_dict.values())[0]
            return group_dict

        result_groups: dict[str, DataFrame] = {}
        for variable, group_df in orig_vars_groups:
            if variable in assignments:
                groupe_name = assignments[variable]["group"]
                group_df["variable"] = groupe_name  # group-name
                group_df["idx"] = assignments[variable]["gr_variable"]  # index
                if groupe_name not in result_groups:
                    # print(f"adding {gn=}")
                    result_groups[groupe_name] = group_df
                else:
                    result_groups[groupe_name] = pd.concat([result_groups[groupe_name], group_df])
            else:
                result_groups[variable] = group_df

        for rg in result_groups.values():
            rg["idx"] = rg["idx"].astype(int)
        if not always_as_dict and len(result_groups) == 1:
            return list(result_groups.values())[0]
        return result_groups

    @staticmethod
    def prepare_single_select(df: DataFrame) -> DataFrame:
        return df.explode("value")

    @staticmethod
    def create_coder_pivot_df(df: DataFrame) -> DataFrame:
        df = df.reset_index()
        index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
        pv_df = df.pivot(columns="user_id", values="value", index=index)
        return pv_df

    @staticmethod
    def calc_agreements(df: DataFrame, agreement_types: list[Literal["gwet", "kappa"]] = ('gwet',)) -> dict[
        str, float]:
        if len(df) == 0:
            return {_: nan for _ in agreement_types}
        pv_df = Agreements.create_coder_pivot_df(df)
        result = {}
        for aggr_type in agreement_types:
            try:
                cac = irrCAC.raw.CAC(pv_df)
                result[aggr_type] = cac.gwet()["est"]["coefficient_value"]
            except (ValueError, ZeroDivisionError):
                result[aggr_type] = nan
        return result

    # this annotated stuff is just experimental...
    def create_init_df(self) -> Annotated[DataFrame, DFAgreementsInitModel]:
        df: DataFrame = self.po.get_annotations_results().raw_annotation_df.copy()
        #df.rename(columns={"category": "variable"},
        #          inplace=True)  # todo fix further up in logic. when reading ls studio response
        df.drop(["ann_id", "platform_id"], axis=1, inplace=True)
        df['date'] = df['ts'].dt.date
        df.set_index(["task_id", "user_id"], inplace=True)
        self._init_df = df
        return df

    def get_init_df(self) -> DataFrame:
        return self._init_df.copy()

    def get_variables_groups(self, variables: list[str]) -> Any:
        pass

    def drop_unfinished_tasks(df_: DataFrame) -> DataFrame:
        df__ = Agreements.base_df(df_)
        return df__.groupby("task_id").filter(
            lambda x: len(x.index.get_level_values(1).unique()) > 1)

    @staticmethod
    def time_move(df_: DataFrame) -> Generator[tuple[date, DataFrame], None, None]:
        for day in sorted(df_.date.unique()):
            yield day, df_[df_["date"] <= day]

    @staticmethod
    def prepare_var(base_df, force_default_: Optional[str] = "NONE"):
        if "variable" in base_df.columns:
            base_df = base_df.drop("variable", axis=1)

        base_df = Agreements.drop_unfinished_tasks(base_df)

        # this part to throw out all with more than x coder. get the times, and only keep the latest 2.
        # df = df.join(df_ts)
        # df = df.sort_values("ts", ascending=False).groupby(level=0).head(max_coders).sort_index()
        if base_df.iloc[0]["type"] == "single":
            base_df = Agreements.prepare_single_select(base_df)
        else:
            pass  # todo
        # df = df.drop("type", axis=1)
        base_df.fillna(force_default_, inplace=True)
        return base_df

        # nice for when having time_move
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

    def agreement_calc(self, variables: list[str],
                       force_default: Optional[str] = "NONE", max_coders: int = 2):
        df = self.get_init_df()

        variables_dfs = self.select_variables(variables)
        for n, df in variables_dfs.items():
            print(n, len(df))

        for var, v_df in variables_dfs.items():
            v_df = Agreements.prepare_var(v_df, force_default).reset_index()
            self.calc_agreements(v_df)

            for day, accum_df in self.time_move(v_df):
                pass
                #print(day, len(accum_df), self.calc_agreements(accum_df))
            # minimal...

            """
            index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
            min_index = index + ["user_id"]
            df_min = v_df.copy().set_index(min_index)
            df_min['position'] = df_min.groupby(level=0).cumcount()
            df_min = df_min.droplevel('user_id', axis=0)
            """

        return df


if __name__ == "__main__":
    po = get_project(43)
    Agreements(po).agreement_calc( ["nature_any", "nature_text", "nature_visual", "nep_material_visual"])
