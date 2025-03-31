from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Any, Iterable, Annotated

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field, ConfigDict, model_validator, PlainSerializer

from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel, TaskResultModel

PlLang = tuple[str, str]
ProjectAccess = int | str | PlLang


def find_name_fixes(orig_keys: Iterable[str],
                    data_extension: "ProjectAnnotationExtension",
                    report_missing: bool = False) -> list[tuple[str, str]]:
    name_fixes: list[tuple[str, str]] = []
    for k in orig_keys:
        item_fix = data_extension.fixes.get(k)
        if item_fix:
            if item_fix.name_fix:
                name_fixes.append((k, item_fix.name_fix))
        elif report_missing:
            print(f"Missing fix for {k}")

    return name_fixes


class Choice(BaseModel):
    value: str
    alias: Optional[str] = None

    @property
    def annot_val(self) -> str:
        if self.alias:
            return self.alias
        return self.value


class Choices(BaseModel):
    name: str
    toName: str
    options: list[Choice]
    choice: Literal["single", "multiple"] = "single"
    indices: Optional[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def create_indices(cls, data: "Choices"):
        data.indices = [c.annot_val for c in data.options]
        return data

    def get_index(self, value: str | list[str]) -> int | list[int]:
        if isinstance(value, str):
            return self.indices.index(value)
        else:
            return [self.indices.index(v) for v in value]

    def insert_option(self, index, choice: Choice):
        self.options.insert(index, choice)
        self.indices = [c.annot_val for c in self.options]

    def raw_options_list(self) -> list[str]:
        return [c.annot_val for c in self.options]


class ResultStruct(BaseModel):
    # project_id: int
    ordered_fields: list[str]
    choices: dict[str, Choices]
    free_text: list[str]
    inputs: dict[str, str] = Field(description="Map from el.name > el.value")

    def apply_extension(self, data_extensions: "ProjectAnnotationExtension", allow_non_existing_defaults: bool = True):

        ordered_name_fixes = find_name_fixes(self.ordered_fields, data_extensions)
        for old, new in ordered_name_fixes:
            self.ordered_fields[self.ordered_fields.index(old)] = new
        choices_name_fixes = find_name_fixes(self.choices.keys(), data_extensions, True)

        for old, new in choices_name_fixes:
            choice = self.choices[old]
            choice.name = new
            self.choices[new] = choice
            del self.choices[old]

        # check if defaults are correct
        for k, v in self.choices.items():
            ext = data_extensions.get_from_rev(v.name)
            # catch non-existing defaults..
            if ext and ext.default:
                if not allow_non_existing_defaults:
                    if v.choice == "single":
                        if not isinstance(ext.default, str):
                            raise ValueError(f"Choice {k} has default value {ext.default}")
                        if ext.default not in (acc_vals := [c.annot_val for c in v.options]):
                            raise ValueError(
                                f"Choice {k} has default invalid value {ext.default}, options: {acc_vals}")
                    elif v.choice == "multiple":
                        if not isinstance(ext.default, list):
                            raise ValueError(f"Choice {k} has default value {ext.default}")
                        if any(d not in (acc_vals := [c.annot_val for c in v.options]) for d in ext.default):
                            raise ValueError(
                                f"Choice {k} has default invalid value {ext.default}, options: {acc_vals}")
                # TODO pass actually add the default...
                v.insert_option(0, Choice(value=ext.default, alias=ext.default))

        text_name_fixes = find_name_fixes(self.free_text, data_extensions, True)
        for old, new in text_name_fixes:
            self.free_text[self.free_text.index(old)] = new

        pass

    def question_type(self, q) -> str:
        """
        in some parts, we turn column indices into $, so this is the
        way to get the original type
        # todo, not good approach. we should have the column merging
        # as a flag and store the original type with it
        :param q:
        :return:
        """
        if "$" in q:
            q = q.replace("$", "0")
        if q in self.choices:
            return self.choices[q].choice
        elif q in self.free_text:
            return "text"
        else:
            raise ValueError("unknown question type", q)


class VariableExtension(BaseModel):
    name_fix: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str | list[str]] = None
    deprecated: Optional[bool] = None


class ProjectAnnotationExtension(BaseModel):
    fixes: dict[str, VariableExtension]
    fix_reverse_map: dict[str, str] = Field(description="fixes[k].name_fix = fixes[k]", default_factory=dict,
                                            exclude=True)

    def model_post_init(self, __context: Any) -> None:
        for k, v in self.fixes.items():
            if v.name_fix:
                self.fix_reverse_map[v.name_fix] = k
            else:
                self.fix_reverse_map[k] = k

    def get_from_rev(self, new_name: str) -> Optional[VariableExtension]:
        orig_name = self.fix_reverse_map.get(new_name)
        if orig_name:
            return self.fixes[orig_name]


class PrincipleRow(BaseModel):
    task_id: int
    ann_id: int
    user_id: int
    # user: Optional[str] = None
    platform_id: str
    ts: datetime
    type: str
    category: str
    value: list[str]


class FullAnnotationRow(BaseModel):
    task_id: int
    ann_id: int
    platform_id: str
    user_id: int
    user: Optional[str] = None
    updated_at: datetime
    results: dict[str, Optional[list[str]]] = Field(default_factory=dict)


class MyProject(BaseModel):
    platform: str
    language: str
    project_data: ProjectModel
    annotation_structure: ResultStruct
    data_extensions: Optional[ProjectAnnotationExtension] = None
    raw_annotation_result: Optional[list[TaskResultModel]] = None
    project_views: Optional[list[ProjectViewModel]] = None
    raw_annotation_df: Optional[pd.DataFrame] = None
    assignment_df: Optional[pd.DataFrame] = None

    _extension_applied: Optional[bool] = False

    @property
    def project_id(self) -> int:
        return self.project_data.id

    def apply_extension(self) -> None:
        if not self.data_extensions:
            print("no data extension set/applied")
            return

        # apply it to the struct...
        if not self._extension_applied:
            self.annotation_structure.apply_extension(self.data_extensions)

        self._extension_applied = True

    def get_annotation_df(self, debug_task_limit: Optional[int] = None) -> tuple[DataFrame, DataFrame]:

        assignment_df_rows = []
        rows = []

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        q_extens = {k: var_method(k, v) for k, v in self.data_extensions.fixes.items()}

        debug_mode = debug_task_limit is not None

        for task in self.raw_annotation_result.annotations:
            # print(task.id)
            for ann in task.annotations:
                # print(f"{task.id=} {ann.id=}")
                if ann.was_cancelled:
                    # print(f"{task.id=} {ann.id=} C")
                    continue
                # print(f"{task.id=} {ann.id=} {len(ann.result)=}")
                assignment_df_rows.append(
                    PrincipleRow.model_construct(task_id=task.id,
                                                 ann_id=ann.id,
                                                 user_id=ann.completed_by,
                                                 ts=ann.updated_at,
                                                 platform_id=task.data["platform_id"]).model_dump(by_alias=True)
                )

                for q_id, question in enumerate(ann.result):
                    new_name = q_extens[question.from_name]
                    if not new_name:
                        continue
                    # print(question)
                    if question.type == "choices":
                        type_ = self.annotation_structure.choices[new_name].choice
                    elif question.type == "textarea":
                        type_ = "text"
                    else:
                        print("unknown question type")
                        type_ = "x"
                    rows.append(PrincipleRow(task_id=task.id,
                                             ann_id=ann.id,
                                             platform_id=task.data["platform_id"],
                                             user_id=ann.completed_by,
                                             ts=ann.updated_at,
                                             category=new_name,
                                             type=type_,
                                             value=question.value.direct_value).model_dump(by_alias=True))

            if debug_mode:
                debug_task_limit -= 1
                if debug_task_limit == 0:
                    break

        df = DataFrame(rows)
        assignment_df = DataFrame(assignment_df_rows)
        """
        raw_annotation_df = df.astype(
            {"task_id": "int32", "ann_id": "int32", 'user_id': "category", "value_idx": "int32",
             'type': "category", 'question': "category", 'value': "string"})
        """
        return df, assignment_df

    def get_annotation_df2(self, debug_task_limit: Optional[int] = None,
                           insert_defaults: bool = True) -> DataFrame:
        """
        this one creates task_id, ann_id rows
        :param debug_task_limit:
        :return:
        """
        rows = []

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        q_extens = {k: var_method(k, v) for k, v in self.data_extensions.fixes.items()}

        debug_mode = debug_task_limit is not None

        for task in self.raw_annotation_result.annotations:
            # print(task.id)
            for ann in task.annotations:
                # print(f"{task.id=} {ann.id=}")
                if ann.was_cancelled:
                    # print(f"{task.id=} {ann.id=} C")
                    continue
                # print(f"{task.id=} {ann.id=} {len(ann.result)=}")

                row_data = {}
                for q_id, question in enumerate(ann.result):
                    new_name = q_extens.get(question.from_name, None)
                    if not new_name:
                        continue
                    row_data[new_name] = question.value.direct_value
                rows.append(FullAnnotationRow(task_id=task.id,
                                              platform_id=task.data.get("platform_id", ""),
                                              ann_id=ann.id,
                                              user_id=ann.completed_by,
                                              updated_at=ann.updated_at,
                                              results=row_data).model_dump(by_alias=True))
            if debug_mode:
                debug_task_limit -= 1
                if debug_task_limit == 0:
                    break

        df = DataFrame(rows)

        self.raw_annotation_df = df.astype(
            {"task_id": "int32", "ann_id": "int32", 'user_id': "category"})
        return self.raw_annotation_df

    def get_default_df(self, df: DataFrame, question: str) -> DataFrame:
        # todo, this should use the reverse map , as we want to work with fixed names from here on
        if "$" in question:
            question = question.replace("$", "0")
        rev_name = self.data_extensions.fix_reverse_map[question]
        if not (question_da := self.data_extensions.fixes.get(rev_name)):
            raise ValueError(f"unknown question: {question} options: {list(self.data_extensions.fixes.keys())}")
        if not (default := question_da.default):
            raise ValueError(f"no default: {question}")

        # Get the valid task_id and ann_id combinations that exist in the original data
        valid_combinations = df[['task_id', 'ann_id', 'user_id']].drop_duplicates()

        # Create a complete DataFrame with valid combinations for the specific question
        complete_df = valid_combinations.copy()
        complete_df['question'] = question

        # Filter the original DataFrame for the specific question
        question_df = df[df["question"] == question].copy()

        # Then merge with question-specific data
        result = pd.merge(
            complete_df,
            question_df[['task_id', 'ann_id', "updated_at", 'question', 'value_idx', 'value']],
            on=['task_id', 'ann_id', 'question'],
            how='left'
        )

        # Fill missing values with default
        result['value'] = result['value'].fillna(default)
        result['value_idx'] = result['value_idx'].fillna(0).astype('int32')

        # Add any other columns needed from the original DataFrame
        if 'type' in df.columns:
            if len(question_df) > 0:
                result['type'] = question_df['type'].iloc[0]  # Use type from the question data
            else:
                type_col = df[df['question'] == question]['type'].iloc[0] if len(
                    df[df['question'] == question]) > 0 else \
                    df['type'].iloc[
                        0]
                result['type'] = type_col
        # todo verify
        if 'updated_at' in df.columns:
            if len(question_df) > 0:
                result['updated_at'] = question_df['updated_at'].iloc[0]  # Use type from the question data
            else:
                type_col = df[df['question'] == question]['updated_at'].iloc[0] if len(
                    df[df['question'] == question]) > 0 else \
                    df['updated_at'].iloc[
                        0]
                result['updated_at'] = type_col

        return result

    def results2csv(self, dest: Path, with_defaults: bool = True, min_coders: int = 1):
        if not with_defaults:
            print("warning, result2csv with_defaults is disabled")
        extra_cols = ["task_id", "num_coders", "users", "cancellations", "updated_at", "username", "displayname",
                      "description"]
        rows = []

        all_fieldnames: list[str] = []
        # task
        for task in self.annotation_results:
            if task.num_coders < min_coders:
                continue
            # annotation
            row_final: dict[str, str | int] = {"num_coders": task.num_coders,
                                               "task_id": task.task_id,
                                               "cancellations": task.num_cancellations,
                                               "users": ", ".join(map(str, task.users))}
            try:
                row_final |= task.data_str(with_defaults)
            except Exception as e:
                print(e)
                raise

            for input_name, input_value in self.annotation_structure.inputs.items():
                # todo, crashed, when direct access. task.data is checked against, config
                row_final[input_name] = task.relevant_input_data.get(input_value)
            for k in row_final:
                if k not in all_fieldnames:
                    all_fieldnames.append(k)
            rows.append(row_final)

        # todo, more robust. collect. what we have.
        final_cols = extra_cols
        for col in self.annotation_structure.ordered_fields:
            final_cols.append(col)

        writer = DictWriter(open(dest, 'w'), fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class UserInfo(BaseModel):
    users: dict[int, str]


class TasksAgreementsChoices(BaseModel):
    values: list[list[str]]


class TasksAgreements(BaseModel):
    choices: list[TasksAgreementsChoices]


class Agreements(BaseModel):
    project_id: int
    # platform: str
    # language: str
    tasks: list[TasksAgreements]
