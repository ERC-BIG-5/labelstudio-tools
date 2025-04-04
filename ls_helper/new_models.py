import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from ls_helper.models.interface_models import InterfaceData, ProjectFieldsExtensions, Choice, Choices, PrincipleRow
from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator, ConfigDict

from ls_helper.models.interface_models import FullAnnotationRow
from ls_helper.my_labelstudio_client.models import ProjectModel as LSProjectModel, ProjectViewModel, TaskResultModel

from ls_helper.settings import SETTINGS, DFCols, DFFormat
from tools.project_logging import get_logger

PlLang = tuple[str, str]
ProjectAccess = int | str | PlLang

logger = get_logger(__file__)


def get_p_access(
        id: Optional[int] = None,
        alias: Optional[str] = None,
        platform: Optional[str] = None,
        language: Optional[str] = None,
) -> ProjectAccess:
    if alias:
        return alias
    elif id:
        return id
    elif platform and language:
        return platform, language
    raise ValueError(f"{id=} {platform=} {language=}, {alias=} not a valid project-access")


class UserInfo(BaseModel):
    users: dict[int, str]

class ProjectCreate(BaseModel):
    title: str
    platform: Optional[str] = "xx"
    language: Optional[str] = "xx"
    description: Optional[str] = None
    alias: Optional[str] = None
    default: Optional[bool] = Field(False, deprecated="default should be on the Overview model")
    label_config_template: Optional[str] = None
    label_config_additions: Optional[list[str]] = Field(default_factory=list)
    coding_game_view_id: Optional[int] = None

    @model_validator(mode="after")
    def post_build(cls, data: "ProjectData") -> "ProjectData":
        if not data.alias:
            data.alias = data.title.lower().replace(" ", "_")
        return data

    @property
    def full_description(self) -> str:
        return f"{self.title}\n{self.platform}:{self.language}\n{self.description}"

    @property
    def pl_lang(self) -> PlLang:
        return self.platform, self.language

    def save(self):
        platforms_overview2.add_project(self)


class ProjectData(ProjectCreate):
    id: int
    _project_data: Optional[LSProjectModel] = None
    _interface_data: Optional[InterfaceData] = None
    _field_extensions: Optional[ProjectFieldsExtensions] = None

    # views, predictions, results

    @property
    def project_data(self) -> LSProjectModel:
        if self._project_data:
            return self._project_data
        fin: Optional[Path] = None
        if (p_i := SETTINGS.projects_dir / f"{self.id}.json").exists():
            fin = p_i
        if not fin:
            raise FileNotFoundError(f"project data file for {self.id}: platform-language does not exist")
        self._project_data = LSProjectModel.model_validate_json(fin.read_text())
        return self._project_data

    @staticmethod
    def parse_label_config_xml(xml_string) -> InterfaceData:
        root: ET.Element = ET.fromstring(xml_string)

        ordered_fields: list[str] = []
        choices = {}
        free_text = []
        input_text_fields: dict[str, str] = {}  # New list for text fields with "$" values

        for el in root.iter():
            if el.tag == "Choices":
                name = el.get('name')
                ordered_fields.append(name)
                # print(choices_element.attrib)
                choice_list = [Choice.model_validate(choice.attrib) for choice in el.findall('./Choice')]
                choices[name] = Choices.model_validate(el.attrib | {"options": choice_list})
            elif el.tag == "TextArea":
                name = el.get('name')
                free_text.append(name)
                ordered_fields.append(name)
            elif el.tag == "Text":
                value = el.get('value')
                if value and value.startswith('$'):
                    name = el.get('name')
                    # keep the $ so we know its a ref to data.
                    input_text_fields[name] = value[1:]
                    ordered_fields.append(name)

        return InterfaceData(
            ordered_fields=ordered_fields,
            orig_choices=choices,
            free_text=free_text,
            inputs=input_text_fields)

    def interface(self,
                  include_text: bool = True) -> InterfaceData:
        """
        caches the structure.
        :param include_text:
        :return:
        """
        if self._interface_data:
            return self._interface_data

        self._interface_data = ProjectData.parse_label_config_xml(self.project_data.label_config)
        return self._interface_data

    @property
    def data_extension(self) -> ProjectFieldsExtensions:
        if self._field_extensions:
            return self._field_extensions
        if (fi := SETTINGS.fixes_dir / "unifixes.json").exists():
            data_extensions = ProjectFieldsExtensions.model_validate(json.load(fi.open()))
        else:
            print(f"no unifixes.json file yet in {SETTINGS.fixes_dir / 'unifix.json'}")
            data_extensions = {}
        if (p_fixes_file := SETTINGS.fixes_dir / f"{self.id}.json").exists():
            p_fixes = ProjectFieldsExtensions.model_validate_json(p_fixes_file.read_text(encoding="utf-8"))

            data_extensions.extensions.update(p_fixes.extensions)
            data_extensions.extension_reverse_map.update(p_fixes.extension_reverse_map)
        self._field_extensions = data_extensions
        return data_extensions

    def get_views(self) -> Optional[list[ProjectViewModel]]:
        view_file = SETTINGS.view_dir / f"{self.id}.json"
        if not view_file.exists():
            return None
        data = json.load(view_file.open())
        return [ProjectViewModel.model_validate(v) for v in data]

    @property
    def view_file(self) -> Optional[Path]:
        return SETTINGS.view_dir / f"{self.id}.json"

    def validate_extensions(self) -> list[str]:
        """
        go through all fixes and mark those, which are not in the structure:
        :return:
        """
        interface = self.interface()
        redundant_extensions = []
        for var in self.data_extension.extensions:
            if var not in interface:
                redundant_extensions.append(var)
                #logger.warning(f"variable from fixes is redundant {var}")
        return redundant_extensions

    def create_annotations_results(self,
                                   accepted_ann_age: Optional[int] = 6) -> "ProjectResult":
        # project_data = p_info.project_data()

        conf = self.interface()
        data_extensions = self.data_extension
        conf.apply_extension(data_extensions)
        mp = ProjectResult(project_data=self)

        from ls_helper.project_mgmt import ProjectMgmt
        mp.raw_annotation_result = ProjectMgmt.get_recent_annotations(mp.id, accepted_ann_age)
        mp.raw_annotation_df, mp.assignment_df = mp.get_annotation_df()
        return mp

    # todo, move somewhere else?
    @staticmethod
    def users() -> "UserInfo":
        pp = Path(SETTINGS.BASE_DATA_DIR / "users.json")
        if not pp.exists():
            users = UserInfo(**{})
            json.dump(users.model_dump(), pp.open("w"), indent=2)
            return users
        else:
            return UserInfo.model_validate(json.load(pp.open()))

class ProjectOverView2(BaseModel):
    projects: dict[ProjectAccess, ProjectData]
    alias_map: dict[str, ProjectData] = Field(default_factory=dict, exclude=True)
    default_map: dict[PlLang, ProjectData] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def create_map(cls, overview: "ProjectOverView2") -> "ProjectOverView2":
        """
        create alias_map and default_map
        """
        for project in overview.projects.values():
            # print(project.id, project.name)
            if project.alias in overview.alias_map:
                print(f"warning: alias {project.alias} already exists")
                continue
            overview.alias_map[project.alias] = project
            pl_l = (project.platform, project.language)

            # is the project has the default flag...
            if project.default:
                # check if the already set default, actually has the flat
                if set_default := overview.default_map.get(pl_l, None):
                    if set_default.default:
                        print(f"warning: default {pl_l} already exists. Not setting {project.title} as default")
                        continue
                overview.default_map[pl_l] = project
            # just set the first pl_l into the default map
            elif pl_l not in overview.default_map:
                overview.default_map[pl_l] = project
        return overview

    @staticmethod
    def load() -> "ProjectOverview2":
        pp = Path(SETTINGS.BASE_DATA_DIR / "projects.json")
        if not pp.exists():
            print("projects file missing")
        # print(pp.read_text())
        # print(ProjectOverView2.model_validate_json(pp.read_text()))
        return ProjectOverView2.model_validate({"projects": json.loads(pp.read_text())})

    def get_project(self, p_access: ProjectAccess) -> ProjectData:
        # int | str | platf_lang_default | platform_lang_name
        if isinstance(p_access, int):
            return self.projects[str(p_access)]
        elif isinstance(p_access, str):
            return self.alias_map[p_access]
        elif isinstance(p_access, tuple) and len(p_access) == 2:
            return self.default_map[p_access]
        raise ValueError(f"unknown project access: {p_access}")

    def add_project(self, p: ProjectCreate, save: bool = True):
        from ls_helper.project_mgmt import ProjectMgmt

        if p.alias in self.alias_map:
            raise ValueError(f"alias {p.alias} already exists")
        if p.default:
            if default_ := self.default_map[(p.platform, p.language)]:
                if default_.default:
                    raise ValueError(f"default {p.pl_lang} already exists")

        project_model, view_model = ProjectMgmt.create_project(p)

        p_i = ProjectData(id=project_model.id, **p.model_dump())

        self.projects[p_i.id] = p_i
        if p_i.default:
            self.default_map[p.pl_lang] = p_i
        self.alias_map[p.alias] = p_i
        if save:
            self.save()

    def save(self):
        projects = {p.id: p for p in self.projects.values()}
        pp = Path(SETTINGS.BASE_DATA_DIR / "projects2.json")
        pp.write_text(json.dumps({id: p.model_dump() for id, p in projects.items()}))


platforms_overview2: ProjectOverView2 = ProjectOverView2.load()


class ProjectResult(BaseModel):
    project_data: ProjectData
    # annotation_structure: ResultStruct
    data_extensions: Optional[ProjectFieldsExtensions] = None
    raw_annotation_result: Optional[list[TaskResultModel]] = None
    project_views: Optional[list[ProjectViewModel]] = None
    raw_annotation_df: Optional[pd.DataFrame] = None
    assignment_df: Optional[pd.DataFrame] = None

    _extension_applied: Optional[bool] = False

    @property
    def id(self) -> int:
        return self.project_data.id

    @property
    def interface(self) -> InterfaceData:
        return self.project_data.interface()

    def get_annotation_df(self, debug_task_limit: Optional[int] = None,
                          drop_cancels: bool = True) -> tuple[DataFrame, DataFrame]:

        assignment_df_rows = []
        rows = []

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        q_extens = {k: var_method(k, v) for k, v in self.project_data.data_extension.extensions.items()}

        debug_mode = debug_task_limit is not None

        for task in self.raw_annotation_result:
            # print(task.id)
            for ann in task.annotations:
                # print(f"{task.id=} {ann.id=}")
                if drop_cancels and ann.was_cancelled:
                    # print(f"{task.id=} {ann.id=} C")
                    continue
                # print(f"{task.id=} {ann.id=} {len(ann.result)=}")
                assignment_df_rows.append(
                    PrincipleRow.model_construct(task_id=task.id,
                                                 ann_id=ann.id,
                                                 user_id=ann.completed_by,
                                                 ts=ann.updated_at,
                                                 platform_id=task.data[DFCols.P_ID]).model_dump(by_alias=True)
                )

                for q_id, question in enumerate(ann.result):
                    new_name = q_extens[question.from_name]
                    if not new_name:
                        continue
                    # print(question)
                    if question.type == "choices":
                        type_ = self.interface.choices[new_name].choice
                    elif question.type == "textarea":
                        type_ = "text"
                    else:
                        print("unknown question type")
                        type_ = "x"
                    rows.append(PrincipleRow(task_id=task.id,
                                             ann_id=ann.id,
                                             platform_id=task.data[DFCols.P_ID],
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
        df.attrs["format"] = DFFormat.raw_annotation
        assignment_df = DataFrame(assignment_df_rows)
        """
        raw_annotation_df = df.astype(
            {"task_id": "int32", "ann_id": "int32", 'user_id': "category", "value_idx": "int32",
             'type': "category", 'question': "category", 'value': "string"})
        """
        return df, assignment_df

    def simplify_single_choices(self, df: DataFrame) -> DataFrame:
        assert df.attrs["format"] == DFFormat.raw_annotation

        result_df = df.copy()

        # Define a function to extract the single value when type is 'single'
        def extract_single_value(row):
            if row['type'] == 'single':
                # Check if value is a list and not empty
                if isinstance(row['value'], list) and len(row['value']) > 0:
                    return row['value'][0]
                # If value is already a string (not a list)
                elif isinstance(row['value'], str):
                    return row['value']
            return None

        # Apply the function to create the new column
        result_df['single_value'] = result_df.apply(extract_single_value, axis=1)
        return result_df

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

        q_extens = {k: var_method(k, v) for k, v in self.data_extensions.extensions.items()}

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
        rev_name = self.data_extensions.extension_reverse_map[question]
        if not (question_da := self.data_extensions.extensions.get(rev_name)):
            raise ValueError(f"unknown question: {question} options: {list(self.data_extensions.extensions.keys())}")
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

    # Simple formatting function that avoids pandas/numpy array checks
    def format_df_for_csv(self, df: DataFrame) -> DataFrame:

        def format_list_for_csv(value_list: list[Any]):
            formatted = []
            for item in value_list:
                # Handle scalars
                if not isinstance(item, list):
                    try:
                        # Check if it's a NaN value using Python's direct check
                        if item != item:  # NaN is the only value that doesn't equal itself
                            formatted.append("")
                        else:
                            formatted.append(str(item))
                    except:
                        formatted.append("")
                else:
                    # Handle lists - join with commas
                    item_str = []
                    for subitem in item:
                        try:
                            if subitem != subitem:  # Check for NaN
                                continue
                            item_str.append(str(subitem))
                        except:
                            continue
                    formatted.append(",".join(item_str))

            return ";".join(formatted)

        formatted_result = df.copy()
        # Apply formatting only to columns that contain lists
        for col in formatted_result.columns:
            if col not in ['task_id', 'platform_id']:
                formatted_result[col] = formatted_result[col].apply(
                    lambda x: format_list_for_csv(x) if isinstance(x, list) else x
                )
        formatted_result.attrs["format"] = DFFormat.flat_csv_ready
        return formatted_result

    def basic_flatten_results(self, min_coders: Optional[int] = 2,
                              column_order: Optional[list[str]] = None) -> DataFrame:
        df = self.raw_annotation_df.copy()

        # Count coders per task and filter to tasks with sufficient coders
        coder_counts = df.groupby('task_id')['user_id'].nunique()
        valid_tasks = coder_counts[coder_counts >= min_coders].index.tolist()
        df = df[df['task_id'].isin(valid_tasks)]

        # Create pivot table with one row per task_id, user_id combination
        # and columns for each category
        flattened_df = df.pivot_table(
            index=['task_id', 'user_id', 'ts', 'platform_id'],
            columns='category',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Reorder columns if specified
        if column_order:
            # Ensure required columns are included
            required_cols = ['task_id', 'user_id', 'ts', 'platform_id']
            ordered_cols = required_cols + [col for col in column_order if col not in required_cols]

            # Add any remaining columns not specified in column_order
            all_cols = flattened_df.columns.tolist()
            final_cols = ordered_cols + [col for col in all_cols if col not in ordered_cols]

            # Apply column ordering, keeping only existing columns
            existing_cols = [col for col in final_cols if col in flattened_df.columns]
            flattened_df = flattened_df[existing_cols]

        return flattened_df

    def flatten_annotation_results(self, min_coders: int = 2, column_order: Optional[list[str]] = None) -> DataFrame:
        df = self.raw_annotation_df.copy()

        # Count coders per task
        coder_counts = df.groupby('task_id')['user_id'].nunique()

        # Filter to only include tasks with at least min_coders
        valid_tasks = coder_counts[coder_counts >= min_coders].index.tolist()

        # Filter the dataframe to only include valid tasks
        df = df[df[DFCols.T_ID].isin(valid_tasks)]

        # Step 1: Create pivot table with task_id and user_id as index
        pivot_df = df.pivot_table(index=[DFCols.T_ID, DFCols.U_ID, DFCols.TS, DFCols.P_ID], columns='category',
                                  values='value',
                                  aggfunc='first').reset_index()

        # Step 2: First group by task_id to get user data in lists
        result = pivot_df.groupby(DFCols.T_ID).apply(
            lambda g: pd.Series({
                # Keep platform_id (they should be the same for a task)
                DFCols.P_ID: g[DFCols.P_ID].iloc[0],
                # For each category column, collect all non-null values in a list
                **{col: g[col].dropna().tolist() for col in g.columns
                   if col not in [DFCols.T_ID, DFCols.U_ID, DFCols.TS, DFCols.P_ID]}
            })
        ).reset_index()

        # Add timestamps as a list ordered by user_id
        result['timestamps'] = pivot_df.groupby(DFCols.T_ID).apply(
            lambda g: g[DFCols.TS].tolist()
        ).values

        # Add user_ids as a list
        result['user_ids'] = pivot_df.groupby('task_id').apply(
            lambda g: g['user_id'].tolist()
        ).values

        result.attrs["format"] = DFFormat.flat
        return result

    def variable_infos(self) -> dict[str, dict[str, Any]]:
        variables = {}

        fixes = self.data_extensions.extensions
        for var, fix_info in fixes.items():
            if new_name := fixes[var].name_fix:
                name = new_name
            else:
                name = var

            if name not in self.interface.ordered_fields:
                continue

            default = fix_info.default
            if name in self.interface.inputs:
                continue
            type = self.interface.question_type(name)
            if type in ["single", "multiple"]:
                options = self.interface.orig_choices[name].raw_options_list()
            else:
                options = []
            variables[name] = {
                "name": name,
                "type": type,
                "options": options,
                "default": default
            }
        return variables

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
