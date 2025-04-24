import logging
import re
import warnings
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, Generator, Literal, Optional

import irrCAC.raw
import pandas as pd
from deprecated.classic import deprecated
from numpy import nan
from pandas import DataFrame
from pydantic import BaseModel, Field

from ls_helper.models.variable_models import ChoiceVariableModel
from tools.project_logging import get_logger

if TYPE_CHECKING:
    from ls_helper.models.main_models import ProjectData


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
    single_overall: Optional[AgreementsCol] = Field(default_factory=dict)
    options_agreements: dict[str, AgreementsCol] = Field(default_factory=dict)
    multi_select_inclusion_agreeement: dict[str, AgreementsCol] = Field(default_factory=dict,
                                                                        description="for multi-select, filtering only those, where at least one coder included the option")


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
        self._assignment_df = self.po.get_annotations_results().assignment_df

        self.results: dict[str, AgreementResult] = {}
        self.collections: dict[str, OptionOccurances] = {}
        self.logger = get_logger(__file__)

        # variable: option: filtered_ids: all_occurrences, agreements, disagreements
        self.option_tasks: dict[str,dict[str, dict[str, list[str]]]] = {}

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
            self,
            variables: list[str],
            always_as_dict: bool = True
    ) -> dict[str, DataFrame] | DataFrame:
        df = self.get_init_df()

        selected_variables = []
        po_variables = self.po.variables()
        variables_dfs: dict[str, DataFrame] = {}

        for var in variables:

            variable_def = po_variables.get(var)
            if not variable_def:
                self.logger.error(f"Unknown Variable: {var}")
                continue
            if not isinstance(variable_def, ChoiceVariableModel):
                self.logger.warning(f"Variable: '{var}' is not a choice variable. Skipping")
                continue

            selected_variables.append(var)

            variables_dfs = df[df["variable"].isin(selected_variables)].groupby(
                "variable"
            )

        if not always_as_dict and len(variables_dfs) == 1:
            return list(variables_dfs.values())[0]
        return dict(list(variables_dfs))

    @staticmethod
    def create_coder_pivot_df(df: DataFrame) -> DataFrame:
        df = df.reset_index()
        index = ["task_id"] + (["idx"] if "idx" in df.columns else [])
        pv_df = df.pivot(columns="user_id", values="value", index=index)
        return pv_df

    @staticmethod
    def _calc_agreements(
            df: DataFrame,
            agreement_types: Agreement_types
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
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            result[aggr_type] = round(
                                _cac.fleiss()["est"]["coefficient_value"], 4
                            )
                    except (ValueError, ZeroDivisionError):
                        result[aggr_type] = nan
                case "gwet":
                    try:
                        if not _cac:
                            _cac = irrCAC.raw.CAC(pv_df)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            result[aggr_type] = round(
                                _cac.gwet()["est"]["coefficient_value"], 4
                            )
                    except (ValueError, ZeroDivisionError):
                        result[aggr_type] = nan
                case "ratio":
                    if not _conflict_count:
                        _conflict_count = pv_df.apply(
                            lambda row: len(row.dropna().unique()) > 1, axis=1
                        ).sum()
                    result[aggr_type] = round(
                        1 - _conflict_count / len(pv_df), 4
                    )
                case "abs":
                    result[aggr_type] = int(len(pv_df) - _conflict_count)
                    result["total"] = int(len(pv_df))
        return result

    # this annotated stuff is just experimental...
    def create_init_df(self) -> Annotated[DataFrame, DFAgreementsInitModel]:
        df: DataFrame = self.po.get_annotations_results(
            self.accepted_ann_age
        ).raw_annotation_df.copy()
        df.drop(["ann_id", "platform_id"], axis=1, inplace=True)
        df["date"] = df["ts"].dt.date
        df.set_index(["task_id", "user_id"], inplace=True)
        self._init_df = df
        return df

    def get_init_df(self) -> DataFrame:
        df_ = self._init_df.copy()
        df_ = self.drop_unfinished_tasks(df_)
        return df_

    def drop_unfinished_tasks(self, df_: DataFrame) -> DataFrame:
        df__ = Agreements.base_df(df_)
        # todo, could also be the project, max_annotation value instead of 1
        reduced = df__.groupby("task_id").filter(
            lambda x: len(x.index.get_level_values(1).unique()) > 1
        )
        task_id_set = lambda df: set(df.index.get_level_values("task_id").unique().tolist())
        df__tasks = task_id_set(df__)
        reduced_tasks = task_id_set(reduced)
        if (num_dropped := (len(df__tasks) - len(reduced_tasks))) > 0:
            self.logger.info(f"dropped unfinished tasks: {num_dropped}")
            if self.logger.level == logging.DEBUG:
                self.logger.debug(f"dropped tasks: {df__tasks - reduced_tasks}")
        return reduced

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
            base_df = base_df.explode("value")
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

    def add_multi_select_default(self, v_df: DataFrame) -> DataFrame:
        ass_df = self._assignment_df.copy()

        ass_df['date'] = pd.to_datetime(ass_df['ts']).dt.date

        # Create a new dataframe with only the necessary columns from df2
        df2_subset = ass_df[['task_id', 'user_id', 'ts', 'date']]

        # Perform an outer merge on task_id and user_id
        merged_df = pd.merge(
            v_df,
            df2_subset,
            on=['task_id', 'user_id'],
            how='outer',
            suffixes=('', '_y')
        )

        # For rows that exist only in df2, fill in default values
        merged_df['idx'] = merged_df['idx'].fillna(0)
        merged_df['type'] = merged_df['type'].fillna('multiple')
        merged_df['value'] = merged_df['value'].fillna('[]')

        # Use timestamps from df1 where available, otherwise from df2
        merged_df['ts'] = merged_df['ts'].combine_first(merged_df['ts_y'])
        merged_df['date'] = merged_df['date'].combine_first(merged_df['date_y'])

        # Drop the extra columns
        merged_df = merged_df.drop(columns=['ts_y', 'date_y'])

        # Sort by task_id and user_id
        merged_df = merged_df.sort_values(['task_id', 'user_id'])
        return merged_df

    def agreement_calc(
            self,
            variables: Optional[list[str]] = None,
            force_default: Optional[str] = "NONE",
            max_coders: int = 2,
            agreement_types: Optional[Agreement_types] = None,
            keep_tasks: bool = True,
    ) -> dict[str, AgreementResult]:
        if not variables:
            variables = list(self.po.choices.keys())
        variables_dfs = self.select_variables(variables)

        if not agreement_types:
            agreement_types = self.agreement_types

        po_variables = self.po.variables()
        for var, v_df in variables_dfs.items():
            print(var, len(v_df))
            result = AgreementResult(variable=var)
            self.results[var] = result
            if "variable" in v_df.columns:
                v_df = v_df.drop("variable", axis=1)

            options = po_variables[var].options
            if v_df.empty:
                continue
            if v_df.iloc[0]["type"] == "single":
                v_df_s = v_df.explode("value")
                v_df_s.fillna(force_default, inplace=True)
                result.single_overall = self._calc_agreements(
                    v_df_s, agreement_types)
                for option in options:
                    # Use vectorized operations when possible for performance
                    option_df = v_df.copy()
                    option_df = option_df.explode("value")
                    option_df = option_df.groupby("task_id").filter(
                        lambda group: (group["value"] == option).any()
                    )
                    result.options_agreements[option] = self._calc_agreements(
                        option_df, agreement_types)

                    # todo....bring in the conflicts.
                    # TODO. the pivot is calculated multiple times, per option?
                    if keep_tasks:
                        #self.collections.setdefault(var, {})[option] = task_ids
                        var_col = self.option_tasks.setdefault(var, {})
                        option_col = var_col.setdefault(option,{})
                        option_col["filtered_ids"]= option_df.index.get_level_values('task_id').unique().tolist()
                        pv_df = Agreements.create_coder_pivot_df(option_df)

                        def match_mask_func(row):
                            # Get non-NaN values
                            non_nan_values = row.dropna().unique()
                            # If all values are the same, there will be only one unique value
                            # (or none if all were NaN)
                            return len(non_nan_values) <= 1

                        # Apply the function to each row
                        match_mask = pv_df.apply(match_mask_func, axis=1)

                        # Split the dataframe
                        option_col["match"] = pv_df[match_mask].index.get_level_values('task_id').unique().tolist()
                        option_col["conflict"] = pv_df[~match_mask].index.get_level_values('task_id').unique().tolist()


            # multi-select
            else:
                v_df = self.add_multi_select_default(v_df)
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
                        option_df, agreement_types)
                    # only the tasks, where one select the option
                    # Group by task_id and check if any user has a value of 1
                    tasks_with_option_select= option_df[option_df['value'] == 1]['task_id'].unique()
                    # Filter the DataFrame to keep only those tasks
                    option_select = option_df[option_df['task_id'].isin(tasks_with_option_select)]
                    result.multi_select_inclusion_agreeement[option] = self._calc_agreements(
                        option_select, agreement_types)

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
    from ls_helper.models.main_models import get_project

    po = get_project(43)
    ag = Agreements(po)
    ag.agreement_calc(
        ["nature_any", "nature_text", "nature_visual", "nep_material_visual"],
        keep_tasks=True,
    )
    pass
