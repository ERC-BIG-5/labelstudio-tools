import re
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, Generator, Literal, Optional

import irrCAC.raw
import pandas as pd
from deprecated.classic import deprecated
from numpy import nan
from pandas import DataFrame
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ls_helper.new_models import ProjectData


# experimental...
class DFAgreementsInitModel(BaseModel):
    task_id: int


Agreement_types = list[Literal["gwet", "fleiss", "ratio", "abs"]]

# calculation-method: value
type AgreementsCol = dict[str, float]
# option: task_id[]
type OptionOccurances = dict[str, list[int]]


class AgreementResult(BaseModel):
    variable: str
    single_overall: Optional[AgreementsCol] = None
    options_agreements: Optional[dict[str, AgreementsCol]] = Field(
        default_factory=dict
    )


class Agreements:
    def __init__(
        self,
        po: "ProjectData",
        accepted_ann_age: int = 6,
        agreement_types: Agreement_types = ("gwet", "ratio", "abs"),
    ) -> None:
        self.po = po
        self.accepted_ann_age = accepted_ann_age
        self.agreement_types = agreement_types
        # group-name: variable-name: index
        self._groups: dict[str, dict[str, int]] = self.find_groups(
            list(po.variable_extensions.extensions.keys())
        )
        self._init_df: Optional[DataFrame] = self.create_init_df()

        self.results: dict[str, AgreementResult] = {}
        self.collections: dict[str, OptionOccurances] = {}

    @staticmethod
    def find_groups(variables):
        pattern = re.compile(r"^(.+)_(\d+)$")

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
        return df.merge(df_in, on="variable", how="left")

    @staticmethod
    def base_df(df: DataFrame) -> DataFrame:
        index_ = df.index.names
        if "task_id" in index_ and "user_id" in index_:
            return df
        else:
            return df.set_index(["task_id", "user_id"])

    def select_variables(
        self, variables: list[str], always_as_dict: bool = True
    ) -> dict[str, DataFrame] | DataFrame:
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
                    {
                        idx_v: {
                            "group": var,
                            "gr_variable": self._groups[var][idx_v],
                        }
                        for idx_v in idx_vars
                    }
                )
                has_groups = True
                # print(assignments)
            else:
                selected_orig_vars.append(var)

        # select all relevant variables
        orig_vars_groups = df[df["variable"].isin(selected_orig_vars)].groupby(
            "variable"
        )

        # if there are no groups we can for convenience also just return the single variable df
        if not has_groups:
            group_dict = {name: group for name, group in orig_vars_groups}
            if not always_as_dict and len(group_dict) == 1:
                return list(group_dict.values())[0]
            return group_dict

        result_groups: dict[str, DataFrame] = {}
        # go through all variables and concat their dataframes when in a group
        for variable, group_df in orig_vars_groups:
            if variable in assignments:
                group_name = assignments[variable]["group"]
                group_df["variable"] = group_name
                group_df["idx"] = assignments[variable]["gr_variable"]
                # set or concat
                if group_name not in result_groups:
                    # print(f"adding {gn=}")
                    result_groups[group_name] = group_df
                else:
                    result_groups[group_name] = pd.concat(
                        [result_groups[group_name], group_df]
                    )
                # todo: TEST THIS APPROACH, remove the outer else, so in case there is none its set...
                # if existing_df := result_groups.get(group_name):
                #     result_groups[group_name] = pd.concat(existing_df, group_df)
                #     continue
            else:
                group_df["idx"] = 0
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
    def _calc_agreements(
        df: DataFrame, agreement_types: Agreement_types
    ) -> AgreementsCol:
        if len(df) == 0:
            return {_: nan for _ in agreement_types}
        pv_df = Agreements.create_coder_pivot_df(df)
        result = {}

        _cac = None
        _conflict_count: Optional[int] = None
        for aggr_type in agreement_types:
            match aggr_type:
                case "kappa":
                    try:
                        if not _cac:
                            _cac = irrCAC.raw.CAC(pv_df)
                        result[aggr_type] = round(
                            _cac.fleiss()["est"]["coefficient_value"], 2
                        )
                    except (ValueError, ZeroDivisionError):
                        result[aggr_type] = nan
                case "gwet":
                    try:
                        if not _cac:
                            _cac = irrCAC.raw.CAC(pv_df)
                        result[aggr_type] = round(
                            _cac.gwet()["est"]["coefficient_value"], 2
                        )
                    except (ValueError, ZeroDivisionError):
                        result[aggr_type] = nan
                case "ratio":
                    if not _conflict_count:
                        _conflict_count = pv_df.apply(
                            lambda row: len(row.dropna().unique()) > 1, axis=1
                        ).sum()
                    result[aggr_type] = round(
                        1 - _conflict_count / len(pv_df), 2
                    )
                case "abs":
                    result[aggr_type] = len(pv_df) - _conflict_count
                    result["total"] = len(pv_df)
        return result

    # this annotated stuff is just experimental...
    def create_init_df(self) -> Annotated[DataFrame, DFAgreementsInitModel]:
        df: DataFrame = self.po.get_annotations_results(
            self.accepted_ann_age
        ).raw_annotation_df.copy()
        # df.rename(columns={"category": "variable"},
        #          inplace=True)  # todo fix further up in logic. when reading ls studio response
        df.drop(["ann_id", "platform_id"], axis=1, inplace=True)
        df["date"] = df["ts"].dt.date
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
            lambda x: len(x.index.get_level_values(1).unique()) > 1
        )

    @staticmethod
    def time_move(
        df_: DataFrame,
    ) -> Generator[tuple[date, DataFrame], None, None]:
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

        return base_df

    def agreement_calc(
        self,
        variables: Optional[list[str]] = None,
        force_default: Optional[str] = "NONE",
        max_coders: int = 2,
        agreement_types: Optional[Agreement_types] = None,
        keep_tasks: bool = False,
    ) -> dict[str, AgreementResult]:
        if not variables:
            variables = list(self.po.choices.keys())
        variables_dfs = self.select_variables(variables)

        if not agreement_types:
            agreement_types = self.agreement_types

        variables = self.po.variables()
        for var, v_df in variables_dfs.items():
            print(var, len(v_df))
            result = AgreementResult(variable=var)
            self.results[var] = result
            if "variable" in v_df.columns:
                v_df = v_df.drop("variable", axis=1)
            v_df = Agreements.drop_unfinished_tasks(v_df)

            options = variables[var].options

            if v_df.iloc[0]["type"] == "single":
                v_df_s = v_df.explode("value")
                v_df_s.fillna(force_default, inplace=True)
                # v_df = Agreements.prepare_var(v_df, force_default).reset_index()
                result.single_overall = self._calc_agreements(
                    v_df_s, agreement_types
                )
                for option in options:
                    # Use vectorized operations when possible for performance
                    option_df = v_df.copy()
                    option_df = option_df.explode("value")
                    option_df = option_df.groupby("task_id").filter(
                        lambda group: (group["value"] == option).any()
                    )
                    result.options_agreements[option] = self._calc_agreements(
                        option_df, agreement_types
                    )
            # multi-select
            else:
                for option in options:
                    option_df = v_df.copy()
                    # Convert to 1/0 values based on option presence
                    option_df["value"] = option_df["value"].apply(
                        lambda x: 1
                        if isinstance(x, list) and option in x
                        else 0
                    )

                    # todo,this also for single selects
                    if keep_tasks:
                        tasks_with_1 = option_df.groupby("task_id")[
                            "value"
                        ].apply(lambda x: (x == 1).any())
                        task_ids = tasks_with_1[tasks_with_1].index
                        self.collections.setdefault(var, {})[option] = task_ids
                    # Calculate agreement for this option
                    result.options_agreements[option] = self._calc_agreements(
                        option_df, agreement_types
                    )

            # for day, accum_df in self.time_move(v_df):
            # pass
            # print(day, len(accum_df), self.calc_agreements(accum_df))
            # minimal...

            """
            index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
            min_index = index + ["user_id"]
            df_min = v_df.copy().set_index(min_index)
            df_min['position'] = df_min.groupby(level=0).cumcount()
            df_min = df_min.droplevel('user_id', axis=0)
            """

        return self.results


if __name__ == "__main__":
    from ls_helper.new_models import get_project

    po = get_project(43)
    ag = Agreements(po)
    ag.agreement_calc(
        ["nature_any", "nature_text", "nature_visual", "nep_material_visual"],
        keep_tasks=True,
    )
    pass
