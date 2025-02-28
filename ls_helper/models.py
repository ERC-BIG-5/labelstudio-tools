import uuid
from collections import Counter
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Any, TypedDict, Iterable

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel


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


class ResultStruct(BaseModel):
    project_id: int
    ordered_fields: list[str]
    choices: dict[str, Choices]
    free_text: list[str]
    inputs: dict[str, str] = Field(description="Map from el.name > el.value")

    def apply_extension(self, data_extensions: "ProjectAnnotationExtension"):

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
                if ext and ext.default:
                    if v.choice == "single":
                        if not isinstance(ext.default, str):
                            raise ValueError(f"Choice {k} has default value {ext.default}")
                        if ext.default not in (acc_vals := [c.annot_val for c in v.options]):
                            raise ValueError(f"Choice {k} has default invalid value {ext.default}, options: {acc_vals}")
                    elif v.choice == "multiple":
                        if not isinstance(ext.default, list):
                            raise ValueError(f"Choice {k} has default value {ext.default}")
                        if any(d not in (acc_vals := [c.annot_val for c in v.options]) for d in ext.default):
                            raise ValueError(f"Choice {k} has default invalid value {ext.default}, options: {acc_vals}")

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
    project_id: int
    fixes: dict[str, VariableExtension]
    fix_reverse_map: dict[str, str] = Field(description="fixes[k].name_fix = fixes[k]", default_factory=dict)

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
    file_path: Path


# modelling LS structure
class ChoicesValue(BaseModel):
    choices: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.choices))


# modelling LS structure
class TextValue(BaseModel):
    text: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.text))


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
    type_: Literal["single", "multiple", "text"] = Field(None, alias="type")
    num_coders: int
    values: list[str] | list[list[str]] = Field(default_factory=list)
    users: list[int | str] = Field(default_factory=list)
    _pre_defaults_added: list[str] | list[list[str]]

    def add(self, value: str | list[str], user_id: int) -> None:
        self.values.append(value)
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
        return "; ".join(self.values)
        # else:
        #     return "; ".join(self._pre_defaults_added)

    def set_predefaults(self):
        self._pre_defaults_added = self.values


class TaskAnnotResults(BaseModel):
    items: Optional[dict[str, TaskAnnotationItem]] = Field(default_factory=dict)
    num_coders: int
    num_cancellations: int
    relevant_input_data: dict[str, Any]
    users: list[int]

    def add(self, item_name: str, value: Any, user_id: int, choices: Optional[Choices] = None) -> None:
        type_ = choices.choice if choices is not None else "text"
        annotation_item = self.items.setdefault(item_name, TaskAnnotationItem.model_validate(
            {"name": item_name, "type": type_, "num_coders": self.num_coders}))

        annotation_item.add(value, user_id)

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
                else:
                    # print(item_name, fix)
                    if fillin_defaults:
                        if value.type_ == "single":
                            # fill it up with default value
                            if len(value.values) != value.num_coders and fix.default:
                                assert isinstance(fix.default, str), "default must be a str"
                                value.values.extend([fix.default] * (value.num_coders - len(value.values)))
                                pass
                        elif value.type_ == "multiple":
                            if len(value.values) != value.num_coders and fix.default:
                                assert isinstance(fix.default, list), "default must be a list"
                                value.values.extend([fix.default] * (value.num_coders - len(value.values)))

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


class ProjectAnnotationResults(BaseModel):
    project_id: int
    annotation_results: list[TaskAnnotResults]

    def apply_extension(self, data_extensions: ProjectAnnotationExtension, fillin_defaults: bool = True) -> None:
        # apply to the results
        for ann_res in self.annotation_results:
            # TaskAnnotResults
            name_fixes: list[tuple[str, str]] = find_name_fixes(ann_res.items.keys(), data_extensions)

            for old, new in name_fixes:
                ann_res.items[new] = ann_res.items[old]
                ann_res.items[new].name = new
                del ann_res.items[old]

        for ann_res in self.annotation_results:
            ann_res.apply_extension(data_extensions, fillin_defaults)


class MyProject(BaseModel):
    project_data: ProjectModel
    annotation_structure: ResultStruct
    raw_annotation_result: Optional[ProjectAnnotations] = None
    calc_annotation_result: Optional[ProjectAnnotationResults] = None
    project_views: Optional[list[ProjectViewModel]] = None
    data_extensions: Optional[ProjectAnnotationExtension] = None

    _extension_applied: Optional[bool] = False

    @property
    def project_id(self) -> int:
        return self.annotation_structure.project_id

    def calculate_results(self):
        if not self.raw_annotation_result:
            print("No raw_annotation_result")
        task_annotation_results = []
        # task
        for task in self.raw_annotation_result.annotations:
            # annotation
            # this should be a model, from which we can also calc the
            users = [ann.completed_by for ann in task.annotations]
            task_calc_results = TaskAnnotResults(num_coders=len(users), relevant_input_data=task.data,
                                                 num_cancellations=task.cancelled_annotations,
                                                 users=users)

            for ann in task.annotations:
                user_id = ann.completed_by
                for ann_res in ann.result:
                    col = ann_res.from_name
                    # all_cols.add(col)
                    if ann_res.type == "choices":
                        choices_c = self.annotation_structure.choices.get(col)
                        if not choices_c:
                            print(f"choice not in result-struct: {col}")
                        else:
                            if choices_c.choice == "single":
                                task_calc_results.add(col, ann_res.str_value, user_id, choices_c)
                            else:
                                task_calc_results.add(col, ann_res.str_value, user_id, choices_c)
                    else:  # text
                        pass
                task_calc_results.set_all_to_pre_default()

            task_annotation_results.append(task_calc_results)

        self.calc_annotation_result = ProjectAnnotationResults(
            project_id=self.project_id,
            annotation_results=task_annotation_results)
        return self.calc_annotation_result

    def apply_extension(self, fillin_defaults: bool = True) -> None:
        if not self.data_extensions:
            print("no data extension set/applied")

        # apply it to the struct...
        if not self._extension_applied:
            self.annotation_structure.apply_extension(self.data_extensions)
        # apply it to the results
        if not self.calc_annotation_result:
            print("No calc_annotation_result")
        else:
            self.calc_annotation_result.apply_extension(self.data_extensions, fillin_defaults=fillin_defaults)

        self._extension_applied = True

    def results2csv(self, dest: Path, with_defaults: bool = True):
        if not with_defaults:
            print("warning, result2csv with_defaults is disabled")
        extra_cols = ["num_coders", "users", "cancellations"]
        rows = []
        # task
        for task in self.calc_annotation_result.annotation_results:
            # annotation
            row_final: dict[str, str | int] = {"num_coders": task.num_coders,
                                               "cancellations": task.num_cancellations,
                                               "users": ", ".join(map(str,task.users))}
            try:
                row_final |= task.data_str(with_defaults)
            except Exception as e:
                print(e)

            for input_name, input_value in self.annotation_structure.inputs.items():
                row_final[input_name] = task.relevant_input_data[input_value]

            rows.append(row_final)

        final_cols = extra_cols
        for col in self.annotation_structure.ordered_fields:
            final_cols.append(col)

        writer = DictWriter(open(dest, 'w'), fieldnames=final_cols)
        writer.writeheader()
        writer.writerows(rows)

    model_config = ConfigDict(validate_assignment=True)
