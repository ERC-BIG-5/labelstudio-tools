import json
import re
import uuid
from collections import Counter
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Any, Iterable

import orjson
from deprecated.classic import deprecated
from pydantic import BaseModel, Field, ConfigDict, RootModel, model_validator

from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel
from ls_helper.settings import SETTINGS


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
        # TODO DOES NOT ACCOUNT FOR SPLITS
        choices_name_fixes = find_name_fixes(self.choices.keys(), data_extensions, True)

        for old, new in choices_name_fixes:
            choice = self.choices[old]
            choice.name = new
            self.choices[new] = choice
            del self.choices[old]

        split_items: list[tuple[str, dict[str, Choices]]] = []
        # check if defaults are correct
        for k, v in self.choices.items():
            ext = data_extensions.get_from_rev(v.name)
            if ext and ext.split_annotation:
                all_new_choices: dict[str, "Choices"] = {}
                for split in ext.split_annotation:
                    new_choices = Choices(name=split.new_name, toName=v.toName, options=[
                        Choice(value=v) for v in split.options
                    ])
                    if split.default:
                        assert split.default in [v.annot_val for v in new_choices.options]
                    all_new_choices[new_choices.name] = new_choices
                split_items.append((k, all_new_choices))
            else:
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

        for old, new_choices in split_items:
            del self.choices[old]
            self.choices |= new_choices
            old_index = self.ordered_fields.index(old)
            self.ordered_fields.remove(old)
            for nc in reversed(new_choices.values()):
                self.ordered_fields.insert(old_index, nc.name)

        text_name_fixes = find_name_fixes(self.free_text, data_extensions, True)
        for old, new in text_name_fixes:
            self.free_text[self.free_text.index(old)] = new

        pass


class VariableSplit(BaseModel):
    new_name: str
    description: Optional[str] = None
    options: list[str]
    default: Optional[str] = None
    value_map: dict[str, str] = None  # From original to new


class VariableExtension(BaseModel):
    name_fix: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str | list[str]] = None
    split_annotation: Optional[list[VariableSplit]] = None
    deprecated: Optional[bool] = None

    def apply_split(self, value: "TaskAnnotationItem", fillin_defaults: bool = True) -> dict[str, "TaskAnnotationItem"]:
        result_values: dict[str, TaskAnnotationItem] = {}
        for split_annotation in self.split_annotation:
            values = [split_annotation.value_map[v] for v in value.values]

            result_values[split_annotation.new_name] = TaskAnnotationItem(
                name=split_annotation.new_name,
                type=value.type_,
                num_coders=value.num_coders,
                values=values,
                _pre_defaults_added=values
            )
            if split_annotation.default and fillin_defaults:
                result_values[split_annotation.new_name].values.extend(
                    [split_annotation.default] * (len(values) - value.num_coders))
        return result_values


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
    project_id: int
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
    created_at: datetime
    updated_at: datetime
    draft_created_at: Optional[datetime] = None
    lead_time: float
    prediction: dict
    result_count: int
    unique_id: uuid.UUID
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
    created_at: datetime
    updated_at: datetime
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Optional[datetime] = None
    project: int
    updated_by: int
    comment_authors: list[int]

    @property
    def num_coders(self) -> int:
        return len(self.annotations)


class TaskAnnotationItem(BaseModel):
    name: str
    type_: Literal["single", "multiple", "text", "datetime"] = Field(None, alias="type")
    num_coders: int
    values: list[list[str]] = Field(default_factory=list)
    value_indices: list[list[int]] = Field(default_factory=list)
    users: list[int | str] = Field(default_factory=list)
    _pre_defaults_added: list[str] | list[list[str]]

    def add(self, value: str | list[str], value_index: int | list[int], user_id: int) -> None:
        self.values.append(value)
        self.value_indices.append(value_index)
        self.users.append(user_id)

    def get_disagreement(self) -> Optional[dict[str, int]]:
        if self.type_ == "single":
            if self.num_coders > 1:
                if len(set(self.values)) > 1:
                    return dict(Counter(self.values))
        else:
            print("[TaskAnnotationItem] no disagreement calculation for other than 'single'")

    def value_str(self, with_defaults: bool = True, with_user: bool = False) -> str:
        # if with_defaults:
        # if with_user:
        #     comb = [f"{v} ({u})" for v, u in zip(self.values, self.users)]
        #     return "; ".join(comb)
        coder_join = [", ".join(cv) for cv in self.values]
        return "; ".join(coder_join)
        # else:
        #     return "; ".join(self._pre_defaults_added)

    def set_predefaults(self):
        self._pre_defaults_added = self.values


class TaskAnnotResults(BaseModel):
    task_id: int
    items: Optional[dict[str, TaskAnnotationItem]] = Field(default_factory=dict)
    num_coders: int
    num_cancellations: int
    relevant_input_data: dict[str, Any]
    users: list[int]

    def add(self, item_name: str, value: str | list[str], value_index: int | list[int], user_id: int,
            type_: Literal["single", "multiple", "text", "datetime"]) -> None:
        annotation_item = self.items.setdefault(item_name, TaskAnnotationItem.model_validate(
            {"name": item_name, "type": type_, "num_coders": self.num_coders}))

        annotation_item.add(value, value_index, user_id)

    def apply_extension(self,
                        annotation_extension: "ProjectAnnotationExtension",
                        fillin_defaults: bool = True) -> None:
        # replace tuple[OLD-NAME: NEW_ITEMS
        split_items: list[tuple[str, dict[str, TaskAnnotationItem]]] = []
        for item_name, value in self.items.items():
            fix = annotation_extension.get_from_rev(item_name)
            if not fix:
                pass
            else:
                if fix.split_annotation:
                    split_items.append((item_name, fix.apply_split(value, fillin_defaults)))
                    continue
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

        for split_old_item_name, new_items in split_items:
            del self.items[split_old_item_name]
            self.items |= new_items

    def get_disagreements(self) -> dict[str, dict[str, int]]:
        """

        Returns {col:str -> {option:str -> count:int}}
        -------

        """
        disagreements: dict[str, dict[str, int]] = {}
        for item_name, item_values in self.items.items():
            if i_d := item_values.get_disagreement():
                disagreements[item_name] = i_d
        return disagreements

    def data(self) -> dict[str, Any]:
        return {k: v.values for k, v in self.items.items()}

    def data_str(self, with_defaults: bool = True) -> dict[str, str]:
        return {k: v.value_str(with_defaults) for k, v in self.items.items()}

    class Config:
        validate_assignment = True

    def set_all_to_pre_default(self) -> None:
        for res in self.items.values():
            res.set_predefaults()


class MyProject(BaseModel):
    platform: str
    language: str
    project_data: ProjectModel
    annotation_structure: ResultStruct
    data_extensions: Optional[ProjectAnnotationExtension] = None
    raw_annotation_result: Optional[ProjectAnnotations] = None
    annotation_results: Optional[list[TaskAnnotResults]] = None
    project_views: Optional[list[ProjectViewModel]] = None

    _extension_applied: Optional[bool] = False

    @property
    def project_id(self) -> int:
        return self.project_data.id

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

            name_fixes = {orig: fix.name_fix for orig, fix in self.data_extensions.fixes.items() if fix.name_fix}
            default_cols = {orig: fix.default for orig, fix in self.data_extensions.fixes.items() if fix.default}
            if task.data["platform_id"] == "1477219043216142337":
                pass
            for ann in task.annotations:
                if ann.was_cancelled:
                    continue
                user_id = ann.completed_by
                #added_cols = set()

                result_dict = {name_fixes.get(ann_res.from_name, ann_res.from_name) : ann_res for ann_res in ann.result}

                for col in name_fixes.values():
                    fix_name = name_fixes.get(col, col)
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
                                default_value_index = self.annotation_structure.choices.get(fix_name).get_index(default_value)
                                task_calc_results.add(fix_name, [default_value], [default_value_index], user_id, type_)
                            else:
                                task_calc_results.add(fix_name, [], [], user_id, type_)
                        else:
                            task_calc_results.add(fix_name, [], [], user_id, "text")

                """
                for ann_res in ann.result:
                    col = ann_res.from_name
                    fix_name = name_fixes.get(col, col)
                    added_cols.add(fix_name)
                    # all_cols.add(col)
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
                for name, default_val in default_cols.items():
                    fix_name = name_fixes.get(name, name)
                    if fix_name in added_cols:
                        continue
                    type_ = self.annotation_structure.choices[name].choice
                    default_value_index = self.annotation_structure.choices.get(name).get_index(default_val)
                    task_calc_results.add(fix_name, [default_val], [default_value_index], user_id, type_)

                for last in name_fixes:
                    if last not in task_calc_results.items:
                        fix_name = name_fixes.get(last, last)
                        type_ = self.annotation_structure.choices.get(fix_name)
                        if type_:
                            type_ = type_.choice
                        else:
                            type_ = "text"
                        task_calc_results.add(fix_name, [], [], user_id, type_)
                """
                # merge all individual indices in the naming ..._ID_... into a one level deper nesting.
                index_names: dict[str, list[tuple[int, str]]] = {}
                pattern = r'_(\d+)_'
                for name, result in task_calc_results.items.items():
                    match = re.search(r'_(\d+)_', name)
                    if match:
                        number = int(match.group(1))  # This will be "123"
                        # print(name, number)
                        group_name = re.sub(pattern, "_", name)
                        # print(group_name)
                        index_names.setdefault(group_name, []).append((number, result.values))

            self.annotation_results.append(task_calc_results)

        return self.annotation_results

    @deprecated
    def apply_extension(self, fillin_defaults: bool = True) -> None:
        if not self.data_extensions:
            print("no data extension set/applied")
            return

        # apply it to the struct...
        if not self._extension_applied:
            self.annotation_structure.apply_extension(self.data_extensions)
        else:
            self.annotation_results.apply_extension(self.data_extensions, fillin_defaults=fillin_defaults)

        self._extension_applied = True

    def results2csv(self, dest: Path, with_defaults: bool = True, min_coders: int = 1):
        if not with_defaults:
            print("warning, result2csv with_defaults is disabled")
        extra_cols = ["num_coders", "users", "cancellations", "updated_at", "username", "displayname", "description"]
        rows = []
        # task
        for task in self.annotation_results:
            if task.num_coders < min_coders:
                continue
            # annotation
            if task.relevant_input_data["platform_id"] == "1477219043216142337":
                pass
            row_final: dict[str, str | int] = {"num_coders": task.num_coders,
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

            rows.append(row_final)

        final_cols = extra_cols
        for col in self.annotation_structure.ordered_fields:
            final_cols.append(col)

        writer = DictWriter(open(dest, 'w'), fieldnames=final_cols)
        writer.writeheader()
        writer.writerows(rows)

    model_config = ConfigDict(validate_assignment=True)


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
