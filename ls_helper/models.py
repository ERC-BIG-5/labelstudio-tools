import json
import re
import uuid
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Any, Iterable, Annotated

import orjson
import pandas as pd
from deprecated.classic import deprecated
from pandas import DataFrame
from pydantic import BaseModel, Field, ConfigDict, RootModel, model_validator, PlainSerializer

from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel
from ls_helper.settings import SETTINGS

# todo bring and import tools,
SerializableDatetime = Annotated[
    datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str, when_used='json')
]

SerializableDatetimeAlways = Annotated[
    datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str, when_used='always')
]


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


class ProjectAnnotations(BaseModel):
    project_id: int  # todo, out...
    annotations: list["TaskResultModel"]
    file_path: Optional[Path] = None


# modelling LS structure
class ChoicesValue(BaseModel):
    choices: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.choices))

    @property
    def direct_value(self) -> list[str]:
        return self.choices


# modelling LS structure
class TextValue(BaseModel):
    text: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.text))

    @property
    def direct_value(self) -> list[str]:
        return self.text


# modelling LS structure
class AnnotationResult(BaseModel):
    id: str
    type: str
    value: ChoicesValue | TextValue
    origin: str
    to_name: str
    from_name: str

    @property
    def str_value(self) -> str:
        return self.value.str_value

    @property
    def direct_value(self) -> list[str]:
        return self.value.direct_value


class TaskAnnotationModel(BaseModel):
    id: int
    completed_by: int
    result: list[AnnotationResult]
    was_cancelled: bool
    ground_truth: bool
    created_at: SerializableDatetimeAlways
    updated_at: SerializableDatetimeAlways
    draft_created_at: Optional[SerializableDatetimeAlways] = None
    lead_time: float
    prediction: dict
    result_count: int
    unique_id: Annotated[uuid.UUID, PlainSerializer(lambda v: str(v), return_type=str, when_used='always')]
    import_id: Optional[int] = None
    last_action: Optional[str] = None
    task: int
    project: int
    updated_by: int
    parent_prediction: Optional[int] = None
    parent_annotation: Optional[int] = None
    last_created_by: Optional[int] = None


# LS structure
class TaskResultModel(BaseModel):
    id: int
    annotations: list[TaskAnnotationModel]
    meta: dict = Field()
    data: dict = Field(..., description="the task data")
    created_at: SerializableDatetimeAlways
    updated_at: SerializableDatetimeAlways
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Optional[SerializableDatetimeAlways] = None
    project: int
    updated_by: int
    comment_authors: list[int]

    @property
    def num_coders(self) -> int:
        return len(self.annotations)


class TaskAnnotationItem(BaseModel):
    name: str
    type_: Literal["single", "multiple", "text", "datetime", "list-single", "list-multiple", "list-text"] = Field(None,
                                                                                                                  alias="type")
    num_coders: int
    values: list[list[str]] = Field(default_factory=list)
    value_indices: list[list[int]] = Field(default_factory=list)
    users: list[int | str] = Field(default_factory=list)

    def add(self, value: str | list[str], value_index: int | list[int], user_id: int) -> None:
        self.values.append(value)
        self.value_indices.append(value_index)
        self.users.append(user_id)

    def value_str(self, with_defaults: bool = True, with_user: bool = False) -> str:
        # if with_defaults:
        # if with_user:
        #     comb = [f"{v} ({u})" for v, u in zip(self.values, self.users)]
        #     return "; ".join(comb)
        if self.type_.startswith("list"):
            coders = []
            for coder_resp in self.values:
                # for item in coder_resp:
                #     items.append(["|".join(item) for item in item])
                coders.append(["|".join(item) for item in coder_resp])
            coder_join = [", ".join(cv) for cv in coders]
        else:
            coder_join = [", ".join(cv) for cv in self.values]
        return "; ".join(coder_join)


class TaskAnnotResults(BaseModel):
    task_id: int
    items: Optional[dict[str, TaskAnnotationItem]] = Field(default_factory=dict)
    num_coders: int
    num_cancellations: int
    relevant_input_data: dict[str, Any]
    users: list[int]

    def add(self, item_name: str, value: list[str] | list[list[str]], value_index: list[int] | list[list[int]],
            user_id: int,
            type_: Literal[
                "single", "multiple", "text", "datetime", "list-single", "list-multiple", "list-text"]) -> None:
        annotation_item = self.items.setdefault(item_name, TaskAnnotationItem.model_validate(
            {"name": item_name, "type": type_, "num_coders": self.num_coders}))

        annotation_item.add(value, value_index, user_id)

    def apply_extension(self,
                        annotation_extension: "ProjectAnnotationExtension",
                        fillin_defaults: bool = True) -> None:
        for item_name, value in self.items.items():
            fix = annotation_extension.get_from_rev(item_name)
            if not fix:
                pass
            else:
                # print(item_name, fix)
                if fillin_defaults:
                    if value.type_ == "single":
                        # fill it up with default value
                        if len(value.values) != value.num_coders and fix.default:
                            assert isinstance(fix.default, str), "default must be a str"
                            value.values.extend([[fix.default] * (value.num_coders - len(value.values))])
                    elif value.type_ == "multiple":
                        if len(value.values) != value.num_coders and fix.default:
                            # assert isinstance(fix.default, list), "default must be a list"
                            value.values.extend([[fix.default] * (value.num_coders - len(value.values))])
        # those that are missing in the row:
        for additional in set(annotation_extension.fixes) - set(self.items):
            fix = annotation_extension.fixes[additional]
            new_name = getattr(fix, "name_fix")
            if not new_name:
                new_name = additional
            if fix.default:
                self.items[new_name] = TaskAnnotationItem(name=new_name, values=[[fix.default]] * self.num_coders,
                                                          num_coders=self.num_coders)

    def data(self) -> dict[str, Any]:
        return {k: v.values for k, v in self.items.items()}

    def data_str(self, with_defaults: bool = True) -> dict[str, str]:
        return {k: v.value_str(with_defaults) for k, v in self.items.items()}

    class Config:
        validate_assignment = True

    def set_all_to_pre_default(self) -> None:
        for res in self.items.values():
            res.set_predefaults()


class PrincipleRow(BaseModel):
    task_id: int
    ann_id: int
    user_id: int
    user: Optional[str] = None
    updated_at: datetime
    type: str = Field(alias="type")
    question: str = Field(alias="question")
    value_idx: int
    value: str


class MyProject(BaseModel):
    platform: str
    language: str
    project_data: ProjectModel
    annotation_structure: ResultStruct
    data_extensions: Optional[ProjectAnnotationExtension] = None
    raw_annotation_result: Optional[ProjectAnnotations] = None
    #annotation_results: Optional[list[TaskAnnotResults]] = Field(None, deprecated="raw_annotation_df")
    project_views: Optional[list[ProjectViewModel]] = None
    raw_annotation_df: Optional[pd.DataFrame] = None

    _extension_applied: Optional[bool] = False

    @property
    def project_id(self) -> int:
        return self.project_data.id

    @deprecated(reason="use straight 2df")
    def calculate_results(self) -> list[TaskAnnotResults]:
        self.annotation_results = []
        for task in self.raw_annotation_result.annotations:
            # annotation
            # this should be a model, from which we can also calc the
            cancel_mask = [ann.was_cancelled for ann in task.annotations]
            users = [ann.completed_by for idx, ann in enumerate(task.annotations) if not cancel_mask[idx]]

            task_calc_results = TaskAnnotResults(num_coders=len(users), relevant_input_data=task.data,
                                                 num_cancellations=task.cancelled_annotations,
                                                 users=users, task_id=task.id)

            all_cos = {orig: fix.name_fix if fix.name_fix else orig for orig, fix in self.data_extensions.fixes.items()}
            default_cols = {orig: fix.default for orig, fix in self.data_extensions.fixes.items() if fix.default}

            for ann in task.annotations:
                if ann.was_cancelled:
                    continue
                user_id = ann.completed_by

                result_dict = {all_cos.get(ann_res.from_name, ann_res.from_name): ann_res for ann_res in ann.result}

                for col in all_cos.values():
                    fix_name = all_cos.get(col, col)
                    if ann_res := result_dict.get(col):
                        if ann_res.type == "choices":
                            choices_c = self.annotation_structure.choices.get(fix_name)
                            value = ann_res.direct_value
                            value_index = self.annotation_structure.choices.get(fix_name).get_index(value)
                            if not choices_c:
                                print(f"choice not in result-struct: {col}")
                            else:
                                task_calc_results.add(fix_name, value, value_index, user_id, choices_c.choice)
                        else:  # text
                            task_calc_results.add(fix_name, ann_res.direct_value, None, user_id, "text")
                    else:
                        default_value = default_cols.get(fix_name)
                        choice = self.annotation_structure.choices.get(fix_name)
                        if choice:
                            type_ = choice.choice
                            if default_value:
                                default_value_index = self.annotation_structure.choices.get(fix_name).get_index(
                                    default_value)
                                task_calc_results.add(fix_name, [default_value], [default_value_index], user_id, type_)
                            else:
                                task_calc_results.add(fix_name, [], [], user_id, type_)
                        else:
                            task_calc_results.add(fix_name, [], [], user_id, "text")

            index_columns_map: dict[str, list[tuple[int, TaskAnnotationItem]]] = {}
            # merge all individual indices in the naming ..._ID_... into a one level deper nesting.
            pattern = r'_(\d+)_|_(\d+)$'
            for name, result in task_calc_results.items.items():
                match = re.search(pattern, name)
                if match:
                    # #todo use the number to order, guarantee right order
                    number = int(match.group(0).strip("_"))
                    group_name = re.sub(pattern, "_", name).strip("_")
                    index_columns_map.setdefault(group_name, []).append((number, result))

            for sum_col, data in index_columns_map.items():
                # delete original columns
                for c in data:
                    del task_calc_results.items[c[1].name]

                for user_idx, user in enumerate(users):
                    values = []
                    values_indices = []
                    for item in data:
                        item_no, item_data = item
                        assert users == item_data.users
                        values.append(item_data.values[user_idx])
                        values_indices.append(item_data.value_indices[user_idx])
                    task_calc_results.add(f"{sum_col}_$", values, values_indices, user, f"list-{item[1].type_}")

            self.annotation_results.append(task_calc_results)
        return self.annotation_results

    def apply_extension(self) -> None:
        if not self.data_extensions:
            print("no data extension set/applied")
            return

        # apply it to the struct...
        if not self._extension_applied:
            self.annotation_structure.apply_extension(self.data_extensions)

        self._extension_applied = True

    def get_annotation_df(self, debug_task_limit: Optional[int] = None) -> DataFrame:
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
                print(f"{task.id=} {ann.id=}")
                if ann.was_cancelled:
                    # print(f"{task.id=} {ann.id=} C")
                    continue
                # print(f"{task.id=} {ann.id=} {len(ann.result)=}")
                rows.append(PrincipleRow.model_construct(task_id=task.id,
                                                         ann_id=ann.id, user_id=ann.completed_by,
                                                         updated_at=ann.updated_at,
                                                         value_idx=-1).model_dump(by_alias=True))
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
                    for v_idx, v in enumerate(question.value.direct_value):
                        rows.append(PrincipleRow(task_id=task.id,
                                                 ann_id=ann.id,
                                                 user_id=ann.completed_by,
                                                 updated_at=ann.updated_at,
                                                 question=new_name,
                                                 type=type_,
                                                 value_idx=v_idx,
                                                 value=v).model_dump(by_alias=True))

            if debug_mode:
                debug_task_limit -= 1
                if debug_task_limit == 0:
                    break

        df = DataFrame(rows)

        self.raw_annotation_df = df.astype(
            {"task_id": "int32", "ann_id": "int32", 'user_id': "category", "value_idx": "int32",
             'type': "category", 'question': "category", 'value': "string"})
        return self.raw_annotation_df

    def get_default_df(self, question: str) -> DataFrame:
        # todo, this should use the reverse map , as we want to work with fixed names from here on
        if "$" in question:
            question = question.replace("$", "0")
        rev_name = self.data_extensions.fix_reverse_map[question]
        if not (question_da := self.data_extensions.fixes.get(rev_name)):
            raise ValueError(f"unknown question: {question} options: {list(self.data_extensions.fixes.keys())}")
        if not (default := question_da.default):
            raise ValueError(f"no default: {question}")
        return self._add_df_defaults(question, default)

    def _add_df_defaults(self, question: str, default: str) -> DataFrame:

        # Get the valid task_id and ann_id combinations that exist in the original data
        df = self.raw_annotation_df
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


class PlatformLanguageOverview(BaseModel):
    id: Optional[int] = None
    coding_game_view_id: Optional[int] = None

    def get_views(self) -> Optional[list[ProjectViewModel]]:
        view_file = SETTINGS.view_dir / f"{self.id}.json"
        if not view_file.exists():
            return None
        data = json.load(view_file.open())
        return [ProjectViewModel.model_validate(v) for v in data]

    def project_data(self) -> ProjectModel:
        platform, lang = ProjectOverview.get_platform_lang_from_id(self.id)
        return ProjectModel.model_validate_json((SETTINGS.projects_dir / f"{platform}/{lang}.json").read_text())

    def get_fixes(self) -> ProjectAnnotationExtension:
        if (fi := SETTINGS.fixes_dir / "unifixes.json").exists():
            data_extensions = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
        else:
            print(f"no unifixes.json file yet in {SETTINGS.fixes_dir / 'unifix.json'}")
            data_extensions = {}
        if (p_fixes_file := SETTINGS.fixes_dir / f"{self.id}.json").exists():
            p_fixes = ProjectAnnotationExtension.model_validate_json(p_fixes_file.read_text(encoding="utf-8"))
            data_extensions.fixes.update(p_fixes.fixes)
            data_extensions.fix_reverse_map.update(p_fixes.fix_reverse_map)

        return data_extensions

    @property
    def platform(self) -> str:
        return platforms_overview.get_platform_lang_from_id(self.id)[0]

    @property
    def language(self) -> str:
        return platforms_overview.get_platform_lang_from_id(self.id)[1]


class ProjectPlatformOverview(RootModel):
    root: dict[str, PlatformLanguageOverview]

    @staticmethod
    def languages() -> list[str]:
        return ["en", "es"]

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, item):
        return self.root[item]

    @staticmethod
    def users() -> "UserInfo":
        pp = Path(SETTINGS.BASE_DATA_DIR / "users.json")
        if not pp.exists():
            users = UserInfo(**{})
            json.dump(users.model_dump(), pp.open("w"), indent=2)
            return users
        else:
            return UserInfo.model_validate(json.load(pp.open()))


ProjectAccess = int | str | tuple[str, str]


@deprecated("reason, we want to use ProjectOverview2 (new_models)")
class ProjectOverview(RootModel):
    root: dict[str, ProjectPlatformOverview]

    @staticmethod
    def platforms() -> list[str]:
        return ["youtube", "twitter", "tiktok", "instagram", "facebook"]

    def __iter__(self):
        for k, v in iter(self.root.items()):
            yield k, v.root

    def __getitem__(self, item: ProjectAccess | str):
        if isinstance(item, str):
            return self.root[item].root
        elif isinstance(item, int):
            return self.get_platform_lang_from_id(item)
        else:
            return self.root[item[0]][item[1]]

    @staticmethod
    def project_data_path(platform: str, language: str) -> Path:
        return SETTINGS.projects_dir / platform / f"{language}.json"

    @staticmethod
    def projects() -> "ProjectOverview":
        pp = Path(SETTINGS.BASE_DATA_DIR / "projects.json")
        if not pp.exists():
            projects = ProjectOverview.model_validate_json(Path("data/projects_template.json").open())
            json.dump(projects.model_dump(), pp.open("w"), indent=2)
            return projects
        else:
            return ProjectOverview.model_validate_json(pp.read_text())

    # todo maybe return Model
    @staticmethod
    def project_data(platform: str, language: str) -> Optional[dict]:
        pp = ProjectOverview.project_data_path(platform, language)
        if not pp.exists():
            return None
        return orjson.loads(pp.open().read())

    @staticmethod
    def get_project_id(platform: str, language: str) -> Optional[int]:
        id = ProjectOverview.projects()[platform][language].id
        if not id:
            raise ValueError(f"No project_id for {platform}, {language}")
        return id

    @staticmethod
    def get_platform_lang_from_id(project_id: int) -> tuple[str, str]:
        for platform, platform_infos in ProjectOverview.projects():
            for lang, project_info in platform_infos.items():
                if project_info.id == project_id:
                    return platform, lang
        raise ValueError(f"No project_id for {project_id}")

    def get_project(self, p_access: ProjectAccess) -> Optional[PlatformLanguageOverview]:
        if isinstance(p_access, int):
            for platform, lang_p in self.root.items():
                for lang, p_data in lang_p.root.items():
                    if p_data.id == p_access:
                        return p_data
        elif isinstance(p_access, tuple):
            platform, language = p_access
            return self[platform][language]

    # todo out
    def get_view_file(self, p_access: ProjectAccess) -> Optional[Path]:
        project = self.get_project(p_access)
        if project:
            return SETTINGS.view_dir / f"{project.id}.json"

    def get_views(self, p_access: ProjectAccess) -> Optional[list[ProjectViewModel]]:
        project = self.get_project(p_access)
        if project:
            return project.get_views()
        root: dict[str, ProjectPlatformOverview]


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


platforms_overview = ProjectOverview.projects()
