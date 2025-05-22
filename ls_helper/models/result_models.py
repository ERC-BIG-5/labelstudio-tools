import json
from logging import Logger
from pathlib import Path
from typing import Optional, Any, cast, TYPE_CHECKING

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from ls_helper.agreements_calculation import Agreements
from ls_helper.models.interface_models import ProjectVariableExtensions, InterfaceData
from ls_helper.models.variable_models import ChoiceVariableModel, VariableModel, VariableType
from ls_helper.my_labelstudio_client.models import ProjectViewModel, TaskResultModel
from ls_helper.settings import SETTINGS, DFCols, DFFormat
from tools.project_logging import get_model_logger
from tools.pydantic_annotated_types import SerializableDatetime

if TYPE_CHECKING:
    from ls_helper.models.main_models import ProjectData


class ProjectResult(BaseModel):
    project_data: "ProjectData"
    data_extensions: Optional[ProjectVariableExtensions] = None
    raw_annotation_result: Optional["ProjectAnnotationResultsModel"] = None
    project_views: Optional[list[ProjectViewModel]] = None
    raw_annotation_df: Optional[pd.DataFrame] = None
    assignment_df: Optional[pd.DataFrame] = None

    _extension_applied: Optional[bool] = False
    _logger: Optional[Logger] = None

    def model_post_init(self, context: Any, /) -> None:
        self._logger = get_model_logger(self)

    @property
    def id(self) -> int:
        return self.project_data.id

    @property
    def interface(self) -> InterfaceData:
        return self.project_data.raw_interface_struct

    def clean_annotation_results(
            self, simplify_single: bool = True, variables: set[str] = None
    ) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
        """

        :param simplify_single:
        :param variables:
        :return: filepath and result-dict: platform_id: [{coder-results}]
        """
        self._logger.info("Building raw annotations dataframe")

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        extension_keys = set(self.project_data.variable_extensions.extensions)
        po_variables = self.project_data.variables(False)
        for var in variables:
            if var not in po_variables:
                raise ValueError(f"{var} not found in project {repr(self.project_data)}")
        q_extens = {
            k: var_method(k, v)
            for k, v in self.project_data.variable_extensions.extensions.items()
        }

        results = {}

        for task in self.raw_annotation_result.task_results:
            task_res = {
                "task_results": []
            }  # , "platform_id": task.data["platform_id"]}
            task_results = task_res["task_results"]
            results[task.data["platform_id"]] = task_results
            for ann in task.annotations:
                ann_result = {}
                if ann.was_cancelled:
                    continue

                for q_id, question in enumerate(ann.result):
                    orig_name = question.from_name
                    if orig_name not in extension_keys:
                        raise ValueError(
                            f"{orig_name} Not found in extensions. Update extension of project {repr(self.project_data)}"
                        )
                    new_name = q_extens.get(question.from_name)

                    if not new_name:
                        self._logger.warning(f"unknown variable... {question.from_name}")
                        continue
                    if variables and new_name not in variables:
                        continue

                    value = question.value.direct_value
                    if (
                            simplify_single
                            and isinstance(
                        po_variables[new_name], ChoiceVariableModel
                    )
                            and cast(
                        ChoiceVariableModel, po_variables[new_name]
                    ).choice
                            == "single"
                    ):
                        value = value[0]
                    ann_result[new_name] = value
                task_results.append(ann_result)

        dest = self.project_data.path_for(
            SETTINGS.annotations_dir,
            alternative=f"clean",
        )
        dest.write_text(json.dumps(results))
        self._logger.info(f"Clean results written to {dest}")
        return dest, results

    def add_default(
            self, variable_def: VariableModel, v_df: DataFrame
    ) -> DataFrame:
        # ass_df["date"] = pd.to_datetime(ass_df["ts"]).dt.date

        # Create a new dataframe with only the necessary columns from df2
        df2_subset = self.assignment_df[
            ["task_id", "ann_id", "ts", "platform_id", "user_id"]
        ]

        # Perform an outer merge on task_id and user_id
        merged_df = pd.merge(
            v_df,
            df2_subset,
            on=["task_id", "ann_id"],
            how="outer",
            suffixes=("", "_y"),
        )

        type_ = variable_def.type
        fillNa = ""
        if type_ == VariableType.choice:
            choice_var = cast(ChoiceVariableModel, variable_def)
            type_ = choice_var.choice
            if type_ == "multiple":
                fillNa = choice_var.default or "[]"
        else:
            type_ = type_.name
        # For rows that exist only in df2, fill in default values
        # todo. instead of 0, we need to fill in the proper length (or 0-4)
        merged_df["variable"] = v_df.name
        merged_df["idx"] = merged_df["idx"].fillna(0)
        merged_df["type"] = merged_df["type"].fillna(type_)
        merged_df["value"] = merged_df["value"].fillna(fillNa)

        # Use timestamps from df1 where available, otherwise from df2
        for col in ["ts", "user_id", "platform_id"]:
            merged_df[col] = merged_df[col].combine_first(
                merged_df[f"{col}_y"]
            )
            merged_df = merged_df.drop(columns=[f"{col}_y"])
        # merged_df["date"] = merged_df["date"].combine_first(
        #     merged_df["date_y"]
        # )

        # Drop the extra columns
        # todo, check landscape-type_text
        # Sort by task_id and user_id
        merged_df = merged_df.sort_values(["task_id", "ann_id"])
        # merged_df = merged_df.set_index(["task_id", "ann_id"])
        return merged_df.reset_index()

    def get_annotation_df(
            self,
            drop_cancels: bool = True,
            fill_defaults: bool = True,
            ignore_groups: bool = False,
            debug_tasks: Optional[list[int]] = None,
            debug_task_limit: Optional[int] = None,
            test_rebuild: bool = False,
    ) -> tuple[DataFrame, DataFrame]:
        """

        :param drop_cancels:
        :param fill_defaults:
        :param ignore_groups:
        :param debug_task_limit:
        :param test_rebuild:
        :return: the value df, the assignment df
        """
        if self.raw_annotation_df is not None and not test_rebuild:
            return self.raw_annotation_df, self.raw_annotation_df
        # todo the value-df still has too many cols, drop them, since we have the assignment df
        self._logger.info("Building raw annotations dataframe")
        assignment_df_rows = []
        rows = []

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        extension_keys = set(self.project_data.variable_extensions.extensions)
        variables = self.project_data.variables(ignore_groups)

        q_extens = {
            k: var_method(k, v)
            for k, v in self.project_data.variable_extensions.extensions.items()
        }

        debug_mode = debug_task_limit is not None

        for task in self.raw_annotation_result.task_results:
            if debug_tasks:
                if task.id not in debug_tasks:
                    continue
            for ann in task.annotations:
                if drop_cancels and ann.was_cancelled:
                    continue

                for q_id, question in enumerate(ann.result):
                    orig_name = question.from_name
                    if orig_name not in extension_keys:
                        raise ValueError(
                            f"{orig_name} Not found in extensions. Update extension of project {repr(self.project_data)}"
                        )
                    new_name = q_extens.get(question.from_name)
                    if not new_name:
                        self._logger.warning(f"unknown variable... {question.from_name}")
                        continue
                    var_def = variables[new_name]
                    idx = 0
                    if var_def.group_name:
                        new_name = var_def.group_name
                        idx = var_def.group_index
                    if question.type == "choices":
                        type_ = cast(
                            ChoiceVariableModel, variables[new_name]
                        ).choice
                    elif question.type == "textarea":
                        type_ = "text"
                    # todo we need to add ranges, from timeline-labels...
                    elif question.type == "timelinelabels":
                        type_ = "range-labels"
                    else:
                        self._logger.warning(f"unknown question type: {q_id},{question}")
                        type_ = "x"
                    rows.append(
                        {
                            "task_id": task.id,
                            "ann_id": ann.id,
                            "platform_id": task.data[
                                DFCols.P_ID
                            ],  # todo just keep it in assignment_df and take it from there
                            "user_id": ann.completed_by,
                            "ts": ann.updated_at,  # todo, same as platform_id
                            "variable": new_name,
                            "idx": idx,
                            "type": type_,
                            "value": question.value.direct_value,
                        }
                    )
                assignment_df_rows.append(
                    {
                        "task_id": task.id,
                        "ann_id": ann.id,
                        "platform_id": task.data[DFCols.P_ID],
                        "user_id": ann.completed_by,
                        "ts": ann.updated_at,
                    }
                )

            if debug_mode:
                debug_task_limit -= 1
                if debug_task_limit == 0:
                    break

        df = DataFrame(rows)

        # pack 'range-labels' which are multiple rows, into one row with lists
        def merge_range_labels(df_) -> DataFrame:
            df_range_labels = df_[df_["type"] == "range-labels"]
            if len(df_range_labels) > 0:
                merge_value = df_range_labels["value"].to_list()
                df_.at[df_range_labels.index[0], "value"] = merge_value
                df_ = df_.drop(df_range_labels.index[1:])
            return df_

        df = df.groupby(["task_id", "ann_id"], as_index=False).apply(merge_range_labels)

        self.assignment_df = DataFrame(assignment_df_rows)
        if fill_defaults:
            df = (
                df.groupby(["variable"], as_index=False)
                .apply(
                    lambda v_df: self.add_default(variables[v_df.name], v_df)
                )
                .reset_index(drop=True)
            )
        # todo, shall we still use this metadata thingy??
        df.attrs["format"] = DFFormat.raw_annotation

        df = df.astype(
            {"task_id": "int32", "ann_id": "UInt8", 'user_id': "UInt8", "idx": "UInt8",
             'type': "category", 'variable': "category", 'value': "object"}, errors='ignore')

        self.raw_annotation_df = df
        return df, self.assignment_df

    def simplify_single_choices(self, df: DataFrame) -> DataFrame:
        assert df.attrs["format"] == DFFormat.raw_annotation
        result_df = df.copy()

        # Define a function to extract the single value when type is 'single'
        def extract_single_value(row):
            if row["type"] == "single":
                # Check if value is a list and not empty
                if isinstance(row["value"], list) and len(row["value"]) > 0:
                    return row["value"][0]
                # If value is already a string (not a list)
                elif isinstance(row["value"], str):
                    return row["value"]
            return None

        # Apply the function to create the new column
        result_df["single_value"] = result_df.apply(
            extract_single_value, axis=1
        )
        return result_df

    # Simple formatting function that avoids pandas/numpy array checks
    def format_df_for_csv(self, df: DataFrame) -> DataFrame:
        def format_list_for_csv(value_list: list[Any]):
            formatted = []
            for item in value_list:
                # Handle scalars
                if not isinstance(item, list):
                    try:
                        # Check if it's a NaN value using Python's direct check
                        if (
                                item != item
                        ):  # NaN is the only value that doesn't equal itself
                            formatted.append("")
                        else:
                            formatted.append(str(item))
                    except Exception:
                        formatted.append("")
                else:
                    # Handle lists - join with commas
                    item_str = []
                    for subitem in item:
                        try:
                            if subitem != subitem:  # Check for NaN
                                continue
                            item_str.append(str(subitem))
                        except Exception:
                            continue
                    formatted.append(",".join(item_str))

            return ";".join(formatted)

        formatted_result = df.copy()
        # Apply formatting only to columns that contain lists
        for col in formatted_result.columns:
            if col not in ["task_id", "platform_id"]:
                formatted_result[col] = formatted_result[col].apply(
                    lambda x: format_list_for_csv(x)
                    if isinstance(x, list)
                    else x
                )
        formatted_result.attrs["format"] = DFFormat.flat_csv_ready
        return formatted_result

    def flatten_annotation_results(
            self, min_coders: int = 2, column_order: Optional[list[str]] = None
    ) -> DataFrame:
        df = self.raw_annotation_df.copy()

        # Count coders per task
        coder_counts = df.groupby("task_id")["user_id"].nunique()

        # Filter to only include tasks with at least min_coders
        valid_tasks = coder_counts[coder_counts >= min_coders].index.tolist()

        # Filter the dataframe to only include valid tasks
        df = df[df[DFCols.T_ID].isin(valid_tasks)]

        # Step 1: Create pivot table with task_id and user_id as index
        pivot_df = df.pivot_table(
            index=[DFCols.T_ID, DFCols.U_ID, DFCols.TS, DFCols.P_ID],
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Step 2: First group by task_id to get user data in lists
        result = (
            pivot_df.groupby(DFCols.T_ID)
            .apply(
                lambda g: pd.Series(
                    {
                        # Keep platform_id (they should be the same for a task)
                        DFCols.P_ID: g[DFCols.P_ID].iloc[0],
                        # For each category column, collect all non-null values in a list
                        **{
                            col: g[col].dropna().tolist()
                            for col in g.columns
                            if col
                               not in [
                                   DFCols.T_ID,
                                   DFCols.U_ID,
                                   DFCols.TS,
                                   DFCols.P_ID,
                               ]
                        },
                    }
                )
            )
            .reset_index()
        )

        # Add timestamps as a list ordered by user_id
        result["timestamps"] = (
            pivot_df.groupby(DFCols.T_ID)
            .apply(lambda g: g[DFCols.TS].tolist())
            .values
        )

        # Add user_ids as a list
        result["user_ids"] = (
            pivot_df.groupby("task_id")
            .apply(lambda g: g["user_id"].tolist())
            .values
        )

        result.attrs["format"] = DFFormat.flat
        return result

    def get_coder_agreements(
            self,
            max_num_coders: int = 2,
            variables: Optional[list[str]] = None,
            exclude_variables: Optional[list[str]] = None,
            gen_csv_tables: bool = True,
    ) -> tuple[list[Path], "Agreements"]:
        from ls_helper.agreements_calculation import Agreements

        ag = Agreements(self)
        ag.agreement_calc(
            variables, exclude_variables, max_coders=max_num_coders
        )

        dest_files = self.project_data.store_agreement_report(
            ag, gen_csv_tables
        )
        self._logger.info(
            f"agreement-results: {[dest.as_posix() for dest in dest_files]}"
        )
        return dest_files, ag

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )


class AnnotationResultStats(BaseModel):
    num_tasks: int
    total_annotations: int
    cancelled_annotations: int


class ProjectAnnotationResultsModel(BaseModel):
    task_results: list[TaskResultModel] = Field(
        ..., description="the task results"
    )
    _stats: Optional[AnnotationResultStats] = None
    dropped_cancellations: Optional[int] = 0
    timestamp: SerializableDatetime

    def stats(self):
        if self._stats:
            return self._stats

        cancelled = 0
        total = 0
        num = len(self.task_results)
        completed = 0
        for t in self.task_results:
            cancelled += t.cancelled_annotations
            total += t.total_annotations
            # todo, this one should come from the project_data
            if t.total_annotations > 1:
                completed += 1
        self._stats = AnnotationResultStats(
            num_tasks=num,
            total_annotations=total,
            cancelled_annotations=cancelled,
        )
        return self._stats

    def completed(self, min_ann: int = 2) -> int:
        return sum(
            1 for t in self.task_results if t.total_annotations >= min_ann
        )

    def drop_cancellations(self) -> "ProjectAnnotationResultsModel":
        canceled = 0
        rea_c = 0
        tasks = []
        for t in self.task_results:
            canceled += t.cancelled_annotations
            ann_new = [ann for ann in t.annotations if not ann.was_cancelled]
            rea_c += len(t.annotations) - len(ann_new)
            tasks.append(
                t.model_copy(
                    update={"annotations": ann_new, "cancelled_annotations": 0}
                )
            )
        return ProjectAnnotationResultsModel(
            task_results=tasks,
            timestamp=self.timestamp,
            dropped_cancellations=canceled,
        )
