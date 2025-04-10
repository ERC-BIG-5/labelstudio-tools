import ast
import json
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from lxml.etree import ElementTree
from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator, ConfigDict
from tqdm import tqdm

from ls_helper.agreements import AgreementReport, export_agreement_metrics_to_csv, analyze_coder_agreement
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.exp.build_configs import LabelingInterfaceBuildConfig, build_from_template
from ls_helper.models.variable_models import VariableModel as FieldModel, ChoiceVariableModel as FieldModelChoice, \
    VariableModel
from ls_helper.models.interface_models import FullAnnotationRow
from ls_helper.models.interface_models import InterfaceData, ProjectVariableExtensions, IChoices, PrincipleRow
from ls_helper.my_labelstudio_client.models import ProjectModel as LSProjectModel, ProjectViewModel, TaskResultModel
from ls_helper.settings import SETTINGS, DFCols, DFFormat
from tools.project_logging import get_logger
from tools.pydantic_annotated_types import SerializableDatetime

PlLang = tuple[str, str]
ProjectAccess = int | str | PlLang | tuple[Optional[int], Optional[str], Optional[str], Optional[str]]

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
        platforms_overview.add_project(self)


class ItemType(str, Enum):
    project_data = auto()
    agreement_report = auto()
    xml_template = auto()


class ProjectData(ProjectCreate):
    id: int
    _project_data: Optional[LSProjectModel] = None
    _interface_data: Optional[InterfaceData] = None
    _variable_extensions: Optional[ProjectVariableExtensions] = None

    # views, predictions, results

    def path_for(self, base_p: Path, alternative: Optional[str] = None, ext: Optional[str] = ".json") -> Path:
        if not ext:
            ext = ".json"
        return base_p / f"{alternative if alternative else self.id}{ext}"

    def __repr__(self) -> str:
        return f"{self.id}/{self.alias}/{self.platform}/{self.language}"

    @property
    def project_data(self) -> LSProjectModel:
        if self._project_data:
            return self._project_data
        fin: Optional[Path] = None
        if (p_i := SETTINGS.projects_dir / f"{self.id}.json").exists():
            fin = p_i
        if not fin:
            raise FileNotFoundError(f"project data file for {self.id}: platform-language does not exist. Call ")
        self._project_data = LSProjectModel.model_validate_json(fin.read_text())
        return self._project_data

    def save_project_data(self, project_data: "LSProjectModel") -> None:
        dest = self.path_for(SETTINGS.projects_dir)
        dest.write_text(project_data.model_dump_json())
        print(f"project-data saved for {repr(self)}: -> {dest}")

    def build_ls_labeling_config(self, alternative_template: Optional[str] = None) -> tuple[Path, ElementTree]:

        template_path = self.path_for(SETTINGS.labeling_templates, alternative_template, ".xml")
        destination_path = self.path_for(SETTINGS.built_labeling_configs, alternative_template, ".xml")

        if not template_path.exists():
            raise FileNotFoundError(f"No template File: {template_path.stem}")
        build_config = LabelingInterfaceBuildConfig(template=template_path)
        built_tree = build_from_template(build_config)
        built_tree.write(destination_path, encoding="utf-8", pretty_print=True)
        print(f"labelstudio xml labeling config written to {destination_path}")
        return self.path_for(SETTINGS.built_labeling_configs), built_tree

    def read_labeling_config(self, alternative_build: Optional[str] = None) -> str:
        return self.path_for(SETTINGS.built_labeling_configs, alternative_build, ".xml").read_text(encoding="utf-8")

    @property
    def fields(self) -> dict[str, VariableModel]:
        variables = {}

        for orig_name, field in self.raw_interface_struct.ordered_fields_map.items():

            field_extension = self.variable_extensions.extensions[orig_name]
            name = self.variable_extensions.name_fixes[orig_name]
            data = {
                "name": name,
                "orig_name": orig_name,
                "type": self.raw_interface_struct.field_type(orig_name),
            }

            if isinstance(field, IChoices):
                field: IChoices
                data["choice"] = field.choice
                data["orig_options"] = field.raw_options_list()
                data["default"] = field_extension.default
                variables[name] = FieldModelChoice.model_validate(data)
            else:
                variables[name] = VariableModel.model_validate(data)

        return variables

    @property
    def choices(self) -> dict[str, FieldModelChoice]:
        return {k: v for k, v in self.fields.items() if isinstance(v, FieldModelChoice)}

    @property
    def raw_interface_struct(self) -> InterfaceData:
        """
        caches the structure.
        :param include_text:
        :return:
        """
        if self._interface_data:
            return self._interface_data

        self._interface_data = parse_label_config_xml(self.project_data.label_config)
        return self._interface_data

    def save_and_log(self,
                     path_dir: Path,
                     data: InterfaceData | ProjectVariableExtensions | Any,
                     alternative: Optional[str] = None,
                     extension: Optional[str] = None):
        p = self.path_for(path_dir, alternative, extension)
        p.write_text(data.model_dump_json())
        logger.info(f"Save {type(data).__name__} to: {p}")

    def save_extensions(self, raw_interf: ProjectVariableExtensions, alternative: Optional[str] = None) -> None:
        self.save_and_log(SETTINGS.var_extensions_dir, raw_interf, alternative)

    @property
    def variable_extensions(self) -> ProjectVariableExtensions:
        if self._variable_extensions:
            return self._variable_extensions
        if (fi := SETTINGS.var_extensions_dir / "unifixes.json").exists():
            extensions = ProjectVariableExtensions.model_validate(json.load(fi.open()))
        else:
            print(f"no unifixes.json file yet in {SETTINGS.var_extensions_dir / 'unifix.json'}")
            extensions = {}
        if (p_fixes_file := SETTINGS.var_extensions_dir / f"{self.id}.json").exists():
            p_fixes = ProjectVariableExtensions.model_validate_json(p_fixes_file.read_text(encoding="utf-8"))

            extensions.extensions.update(p_fixes.extensions)
            extensions.extension_reverse_map.update(p_fixes.extension_reverse_map)
        else:
            logger.error(
                f"{repr(self)} has no 'variable_extensions' file. Call: 'generate_variable_extensions_template'")
        self._variable_extensions = extensions
        return extensions

    def get_views(self) -> Optional[list[ProjectViewModel]]:
        view_file = self.path_for(SETTINGS.view_dir)
        if not view_file.exists():
            return None
        data = json.load(view_file.open())
        return [ProjectViewModel.model_validate(v) for v in data]

    @property
    def view_file(self) -> Optional[Path]:
        return SETTINGS.view_dir / f"{self.id}.json"

    def validate_extensions(self) -> list[str]:
        """
        go through all extensions and mark those, which are not in the structure:
        :return:
        """
        interface = self.raw_interface_struct
        redundant_extensions = []
        for var in self.variable_extensions.extensions:
            if var not in interface:
                redundant_extensions.append(var)
                logger.info(f"variable from extensions is redundant {var}")
        return redundant_extensions

    def get_raw_annotations_results(self,
                                    accepted_ann_age: Optional[int] = 6) -> "ProjectAnnotationResultsModel":
        # project_data = p_info.project_data()
        data_extensions = self.variable_extensions
        # todo should happen inside that function or when post_init
        self.raw_interface_struct.apply_extension(data_extensions)
        from ls_helper.project_mgmt import ProjectMgmt
        _, raw_annotation_result = ProjectMgmt.get_recent_annotations(self.id, accepted_ann_age)
        return raw_annotation_result

    def get_annotations_results(self, accepted_ann_age: Optional[int] = 6) -> "ProjectResult":
        from ls_helper.project_mgmt import ProjectMgmt
        mp = ProjectResult(project_data=self)
        from_existing, mp.raw_annotation_result = ProjectMgmt.get_recent_annotations(mp.id, accepted_ann_age)
        if from_existing:
            raw_df_file = SETTINGS.annotations_dir / f"raw_{self.id}.pickle"
            if raw_df_file.exists():
                mp.raw_annotation_df = pd.read_pickle(raw_df_file)
                mp.assignment_df = pd.read_pickle(SETTINGS.annotations_dir / f"ass_{self.id}.pickle")
                # this, cuz values are lists.
                # mp.raw_annotation_df['value'] = mp.raw_annotation_df['value'].apply(ast.literal_eval)
                # mp.raw_annotation_df['platform_id'] = mp.raw_annotation_df['platform_id'].astype(str)
                return mp
        # new file or there is no raw_dataframe
        mp.raw_annotation_df, mp.assignment_df = mp.get_annotation_df()
        mp.raw_annotation_df.to_pickle(SETTINGS.annotations_dir / f"raw_{self.id}.pickle")
        mp.assignment_df.to_pickle(SETTINGS.annotations_dir / f"ass_{self.id}.pickle")
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

    def store_agreement_report(self, agreement_report: AgreementReport, gen_csv_tables: bool = True):
        (p := self.path_for(ItemType.agreement_report)).write_text(
            agreement_report.model_dump_json(exclude_none=True, indent=2))
        print(f"agreement_report -> {p.as_posix()}")
        if gen_csv_tables:
            p_csv = self.path_for(ItemType.agreement_report, ".csv")
            rows = [dict(r) for r in export_agreement_metrics_to_csv(agreement_report, p_csv)]
            print(f"agreement_report -> {p_csv.as_posix()}")
            df = DataFrame(rows)
            df = df[df["option"] == "VARIABLE_LEVEL"]
            p_csv = self.path_for(ItemType.agreement_report, "_vars.csv")
            df.to_csv(p_csv)
            print(f"agreement_report -> {p_csv.as_posix()}")

        return p

    def get_agreement_data(self) -> AgreementReport:
        return AgreementReport.model_validate_json(self.path_for(ItemType.agreement_report).read_text())


class ProjectOverView(BaseModel):
    projects: dict[ProjectAccess, ProjectData]
    alias_map: dict[str, ProjectData] = Field(default_factory=dict, exclude=True)
    default_map: dict[PlLang, ProjectData] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def create_map(cls, overview: "ProjectOverView") -> "ProjectOverView":
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
        return ProjectOverView.model_validate({"projects": json.loads(pp.read_text())})

    def get_project(self, p_access: ProjectAccess) -> ProjectData:
        # int | str | platf_lang_default | platform_lang_name
        if isinstance(p_access, int):
            return self.projects[str(p_access)]
        elif isinstance(p_access, str):
            return self.alias_map[p_access]
        elif (is_t := isinstance(p_access, tuple)) and (l := len(p_access)) == 2:
            return self.default_map[p_access]
        elif is_t and l == 4:
            id, alias, platform, language = p_access
            if alias:
                return self.alias_map[alias]
            elif id:
                return self.projects[str(id)]
            elif platform and language:
                return self.default_map[(platform, language)]
            raise ValueError(
                f"\n\nYour project parameters where not right: {id=} {platform=} {language=}, {alias=} not a valid project-access")
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
        # todo get the label_config: will be <View></View> into the ls_data.labeling_configs/template folder

    def save(self):
        projects = {p.id: p for p in self.projects.values()}
        pp = Path(SETTINGS.BASE_DATA_DIR / "projects2.json")
        pp.write_text(json.dumps({id: p.model_dump() for id, p in projects.items()}))


platforms_overview: ProjectOverView = ProjectOverView.load()


def get_project(id: Optional[int] = None,
                alias: Optional[str] = None,
                platform: Optional[str] = None,
                language: Optional[str] = None) -> ProjectData:
    po = platforms_overview.get_project((id, alias, platform, language))
    logger.info(repr(po))
    return po


class ProjectResult(BaseModel):
    project_data: ProjectData
    # annotation_structure: ResultStruct
    data_extensions: Optional[ProjectVariableExtensions] = None
    raw_annotation_result: Optional["ProjectAnnotationResultsModel"] = None
    project_views: Optional[list[ProjectViewModel]] = None
    raw_annotation_df: Optional[pd.DataFrame] = None
    assignment_df: Optional[pd.DataFrame] = None

    _extension_applied: Optional[bool] = False

    @property
    def id(self) -> int:
        return self.project_data.id

    @property
    def interface(self) -> InterfaceData:
        return self.project_data.raw_interface_struct

    def get_annotation_df(self, debug_task_limit: Optional[int] = None,
                          drop_cancels: bool = True) -> tuple[DataFrame, DataFrame]:
        logger.info(f"Building raw annotaions dataframe")
        assignment_df_rows = []
        rows = []

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        extension_keys = set(self.project_data.variable_extensions.extensions)

        q_extens = {k: var_method(k, v) for k, v in self.project_data.variable_extensions.extensions.items()}

        debug_mode = debug_task_limit is not None

        for task in tqdm(self.raw_annotation_result.task_results):
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
                    orig_name = question.from_name
                    if orig_name not in extension_keys:
                        raise ValueError(
                            f"{orig_name} Not found in extensions. Update extension of project {repr(self.project_data)}")
                    new_name = q_extens.get(question.from_name)
                    if not new_name:
                        continue
                    # print(question)
                    if question.type == "choices":
                        type_ = self.project_data.fields[new_name].choice
                    elif question.type == "textarea":
                        type_ = "text"
                    else:
                        print("unknown question type")
                        type_ = "x"
                    rows.append(PrincipleRow.model_construct(task_id=task.id,
                                             ann_id=ann.id,
                                             platform_id=task.data[DFCols.P_ID],
                                             user_id=ann.completed_by,
                                             ts=ann.updated_at,
                                             variable=new_name,
                                             type=type_,
                                             value=question.value.direct_value).model_dump(by_alias=True))

            if debug_mode:
                debug_task_limit -= 1
                if debug_task_limit == 0:
                    break

        df = DataFrame(rows)
        # todo, shall we still use this metadata thingy??
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

    def get_coder_agreements(self, min_num_coders: int = 2, variables: Optional[list[str]] = None,
                             gen_csv_tables: bool = True) -> tuple[
        Path, AgreementReport]:
        agreement_report = analyze_coder_agreement(self.raw_annotation_df, self.assignment_df,
                                                   self.project_data.choices, min_num_coders, variables)
        dest = self.project_data.store_agreement_report(agreement_report, gen_csv_tables)
        return dest, agreement_report

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class AnnotationResultStats(BaseModel):
    num_tasks: int
    total_annotations: int
    cancelled_annotations: int


class ProjectAnnotationResultsModel(BaseModel):
    task_results: list[TaskResultModel] = Field(..., description="the task results")
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
        self._stats = AnnotationResultStats(num_tasks=num,
                                            total_annotations=total,
                                            cancelled_annotations=cancelled)
        return self._stats

    def completed(self, min_ann: int = 2) -> int:
        return sum(1 for t in self.task_results if t.total_annotations >= min_ann)

    def drop_cancellations(self) -> "ProjectAnnotationResultsModel":
        canceled = 0
        rea_c = 0
        tasks = []
        for t in self.task_results:
            canceled += t.cancelled_annotations
            ann_new = [ann for ann in t.annotations if not ann.was_cancelled]
            rea_c += len(t.annotations) - len(ann_new)
            tasks.append(t.model_copy(update={"annotations": ann_new,
                                              "cancelled_annotations": 0}))
        return ProjectAnnotationResultsModel(task_results=tasks,
                                             timestamp=self.timestamp,
                                             dropped_cancellations=canceled)
