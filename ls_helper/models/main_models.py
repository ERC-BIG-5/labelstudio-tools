import json
import re
from csv import DictWriter
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast, Annotated

import pandas as pd
from lxml.etree import ElementTree
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ls_helper.agreements_calculation import AgreementResult
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.exp.build_configs import (
    LabelingInterfaceBuildConfig,
    build_from_template,
)
from ls_helper.funcs import get_latest_annotation_file
from ls_helper.models.interface_models import (
    IChoices,
    InterfaceData,
    ProjectVariableExtensions,
)
from ls_helper.models.variable_models import (
    VariableModel,
    ChoiceVariableModel,
    VariableType,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectModel as LSProjectModel,
    ProjectModel,
    ProjectViewCreate,
)
from ls_helper.my_labelstudio_client.models import (
    ProjectViewModel,
    TaskResultModel,
)
from ls_helper.my_labelstudio_client.models import (
    Task as LSTask,
)
from ls_helper.my_labelstudio_client.models import (
    TaskList as LSTaskList,
)
from ls_helper.settings import (
    SETTINGS,
    DFCols,
    DFFormat,
    ls_logger,
    TIMESTAMP_FORMAT,
)
from tools.files import save_json, read_data
from tools.project_logging import get_logger
from tools.pydantic_annotated_types import SerializableDatetime

if TYPE_CHECKING:
    from ls_helper.agreements_calculation import Agreements

PlLang = tuple[str, str]
ProjectAccess = (
    int
    | str
    | PlLang
    | tuple[Optional[int], Optional[str], Optional[str], Optional[str]]
)

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
    raise ValueError(
        f"{id=} {platform=} {language=}, {alias=} not a valid project-access"
    )


class UserInfo(BaseModel):
    users: dict[int, str]


class ProjectCreate(BaseModel):
    title: str
    alias: str
    platform: Optional[str] = "xx"
    language: Optional[str] = "xx"
    description: Optional[str] = None
    default: Optional[bool] = Field(
        False,  # deprecated="default should be on the ProjectOverview model"
    )
    coding_game_view_id: Optional[int] = None

    # todo not sure if this is still needed, sinec alias is required now
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


class ItemType(str, Enum):
    project_data = auto()
    agreement_report = auto()
    xml_template = auto()


class ProjectData(ProjectCreate):
    id: int
    _project_data: Optional[LSProjectModel] = None
    _interface_data: Optional[InterfaceData] = None
    _variable_extensions: Optional[ProjectVariableExtensions] = None
    _ann_results: Optional["ProjectResult"] = None

    # views, predictions, results

    def path_for(
        self,
        base_p: Path,
        alternative: Optional[str] = None,
        ext: Optional[str] = ".json",
    ) -> Path:
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
            raise FileNotFoundError(
                f"project data file for {self.id}: does not exist. Call 'download-project-data'"
            )
        self._project_data = LSProjectModel.model_validate_json(
            fin.read_text()
        )
        return self._project_data

    def save_project_data(self, project_data: "LSProjectModel") -> None:
        dest = self.path_for(SETTINGS.projects_dir)
        dest.write_text(
            project_data.model_dump_json(
                exclude={"control_weights", "parsed_label_config"}
            )
        )
        print(f"project-data saved for {repr(self)}: -> {dest}")

    def build_ls_labeling_config(
        self, alternative_template: Optional[str] = None
    ) -> tuple[Path, ElementTree, bool]:
        """

        :param alternative_template:
        :return: the path of the file, the tree, if its valid
        """
        template_path = self.path_for(
            SETTINGS.labeling_templates, alternative_template, ".xml"
        )
        destination_path = self.path_for(
            SETTINGS.built_labeling_configs, alternative_template, ".xml"
        )
        destination_path.parent.mkdir(exist_ok=True)

        if not template_path.exists():
            raise FileNotFoundError(f"No template File: {template_path.stem}")
        build_config = LabelingInterfaceBuildConfig(template=template_path)
        built_tree, broken_refs, duplicates = build_from_template(build_config)
        built_tree.write(destination_path, encoding="utf-8", pretty_print=True)
        print(
            f"labelstudio xml labeling config written to file://{destination_path.absolute()}"
        )
        valid = not broken_refs and not duplicates
        return (
            self.path_for(SETTINGS.built_labeling_configs),
            built_tree,
            valid,
        )

    def read_labeling_config(
        self, alternative_build: Optional[str] = None
    ) -> str:
        """
        reads the built labeling config file
        :param alternative_build:
        :return:
        """
        return self.path_for(
            SETTINGS.built_labeling_configs, alternative_build, ".xml"
        ).read_text(encoding="utf-8")

    def group_index_variable(self, variable_name: str) -> tuple[str, int]:
        """
        returns the group-name, index
        :param variable_name:
        :return:
        """
        pattern = re.compile(r"^(.+)_(\d+)(?:_(.*))?$")

        match = re.match(pattern, variable_name)
        if not match:
            return variable_name, 0

        # MATCH! kick out None
        match_groups = list(filter(lambda g: g, match.groups()))

        var_strings = []
        index_string = []

        for m in match_groups:
            if m.isnumeric():
                index_string.append(int(m))
            else:
                var_strings.append(m)
        return "_".join(var_strings), index_string[0]

    def variables(
        self, ignore_groups: bool = False
    ) -> dict[str, VariableModel]:
        variables = {}

        # initial basics
        for (
            orig_name,
            field,
        ) in self.raw_interface_struct.ordered_fields_map.items():
            field_extension = self.variable_extensions.extensions[orig_name]
            name = self.variable_extensions.name_fixes[orig_name]
            data = {
                "name": name,
                "orig_name": orig_name,
                "type": self.raw_interface_struct.field_type(orig_name),
            }

            model_class = (
                ChoiceVariableModel
                if isinstance(field, IChoices)
                else VariableModel
            )

            # choices!
            if model_class == ChoiceVariableModel:
                data["choice"] = field.choice
                data["orig_options"] = field.raw_options_list()
                data["default"] = field_extension.default

            variable: VariableModel = model_class.model_validate(data)
            variables[name] = variable

            if ignore_groups:
                continue
            group_name, index = self.group_index_variable(name)
            if group_name == name:
                continue

            variable.group_name = group_name
            variable.group_index = index

            if group_name not in variables:
                group_data = data
                group_data["name"] = group_name
                group_data["group_variables"] = [orig_name]
                variables[group_name] = model_class.model_validate(group_data)
            else:
                if not variables[group_name].group_variables:
                    raise ValueError(
                        f"Error, group-name is already taken by non-group variable: {group_name}. fix non-group variable in extension (project: {self.id})"
                    )

                assert len(variables[group_name].group_variables) > 0, (
                    "group name is already taken by regular variable"
                )
                variables[group_name].group_variables.append(orig_name)

        return variables

    @property
    def variables_names(self) -> list[str]:
        return list(self.variables().keys())

    @property
    def choices(self) -> dict[str, ChoiceVariableModel]:
        return {
            k: v
            for k, v in self.variables().items()
            if isinstance(v, ChoiceVariableModel)
        }

    @property
    def raw_interface_struct(self) -> InterfaceData:
        """
        caches the structure.
        :return:
        """
        if self._interface_data:
            return self._interface_data
        self._interface_data = parse_label_config_xml(
            self.project_data.label_config
        )
        return self._interface_data

    def save_and_log(
        self,
        path_dir: Path,
        data: InterfaceData | ProjectVariableExtensions | Any,
        alternative: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        p = self.path_for(path_dir, alternative, extension)
        p.write_text(data.model_dump_json())
        logger.info(f"Save {type(data).__name__} to: {p}")

    def save_extensions(
        self,
        raw_interf: ProjectVariableExtensions,
        alternative: Optional[str] = None,
    ) -> None:
        self.save_and_log(SETTINGS.var_extensions_dir, raw_interf, alternative)

    @property
    def variable_extensions(self) -> ProjectVariableExtensions:
        if self._variable_extensions:
            return self._variable_extensions

        if (
            p_fixes_file := SETTINGS.var_extensions_dir / f"{self.id}.json"
        ).exists():
            extensions = ProjectVariableExtensions.model_validate_json(
                p_fixes_file.read_text(encoding="utf-8")
            )

            extensions.extension_reverse_map.update(
                extensions.extension_reverse_map
            )
            return extensions
        else:
            raise ValueError(
                f"{repr(self)} has no 'variable_extensions' file. Call: 'generate_variable_extensions_template'"
            )

    def get_views(self) -> Optional[list[ProjectViewModel]]:
        view_file = self.path_for(SETTINGS.view_dir)
        if not view_file.exists():
            return None
        data = json.load(view_file.open())
        return [ProjectViewModel.model_validate(v) for v in data]

    def create_view(
        self, create: ProjectViewCreate, default_hidden_columns: bool = True
    ) -> ProjectViewModel:
        if not create.data.hiddenColumns:
            create.data.hiddenColumns = read_data(
                SETTINGS.BASE_DATA_DIR
                / "default/ls_project_view_hiddenColumns.json"
            )
        return ls_client().create_view(create)

    def refresh_views(self) -> list[ProjectViewModel]:
        views = ls_client().get_project_views(self.id)
        self.path_for(SETTINGS.view_dir).write_text(
            json.dumps([v.model_dump() for v in views])
        )
        return views

    def get_recent_annotations(
        self,
        accepted_age: int,
        use_existing: bool = False,
    ) -> Optional[
        tuple[
            Annotated[bool, "use_local"],
            Optional["ProjectAnnotationResultsModel"],
        ]
    ]:
        """

        :param project_id:
        :param accepted_age:
        :return: true, list of anns; True if existing file
        """
        # todo change the list back to another model in order to pack some functions like dropping cancelations...
        latest_file = get_latest_annotation_file(self.id)
        if latest_file is not None:
            file_dt = datetime.strptime(latest_file.stem, "%Y%m%d_%H%M")
            # print(file_dt, datetime.now(), datetime.now() - file_dt)
            if (
                datetime.now() - file_dt < timedelta(hours=accepted_age)
                or use_existing
            ):
                ls_logger.info(
                    f"Get recent, gets latest annotation: {file_dt:%m%d_%H%M}"
                )

                annotation_file = get_latest_annotation_file(self.id)
                if not annotation_file:
                    ls_logger.warning("No annotation file?!")
                    return None
                task_results = [
                    TaskResultModel.model_validate(t)
                    for t in json.load(annotation_file.open(encoding="utf-8"))
                ]

                return True, ProjectAnnotationResultsModel(
                    task_results=task_results, timestamp=file_dt
                )

        # todo this stuff is old. needs refactoring love. move to ProjectData model
        print("downloading annotations")
        result = ls_client().get_project_annotations(self.id)
        if not result:
            return False, None
        ts = datetime.now()
        res_path = (
            SETTINGS.annotations_dir
            / str(self.id)
            / f"{ts.strftime(TIMESTAMP_FORMAT)}.json"
        )
        res_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"dumping project annotations to {res_path}")
        pa = ProjectAnnotationResultsModel(task_results=result, timestamp=ts)
        json.dump(
            [r.model_dump() for r in result],
            res_path.open("w", encoding="utf-8"),
        )
        return False, pa

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

    def get_raw_annotations_results(
        self, accepted_ann_age: Optional[int] = 6
    ) -> "ProjectAnnotationResultsModel":
        # project_data = p_info.project_data()
        data_extensions = self.variable_extensions
        # todo should happen inside that function or when post_init
        self.raw_interface_struct.apply_extension(data_extensions)

        _, raw_annotation_result = self.get_recent_annotations(
            accepted_ann_age
        )
        return raw_annotation_result

    def fetch_annotations(self) -> "ProjectAnnotationResultsModel":
        mp = ProjectResult(project_data=self)
        _, mp.raw_annotation_result = self.get_recent_annotations(0)
        return mp.raw_annotation_result

    def get_annotations_results(
        self,
        accepted_ann_age: Optional[int] = 6,
        use_existing: bool = False,
    ) -> "ProjectResult":
        """
        :param accepted_ann_age:
        :param use_existing: False, will always be recalculated and not stored.
        :return:
        """
        if self._ann_results:
            return self._ann_results

        ann_results = ProjectResult(project_data=self)
        from_existing, ann_results.raw_annotation_result = (
            self.get_recent_annotations(accepted_ann_age, use_existing)
        )
        if from_existing:
            raw_df_file = SETTINGS.annotations_dir / f"raw_{self.id}.pickle"
            if raw_df_file.exists():
                ann_results.raw_annotation_df = pd.read_pickle(raw_df_file)
                ann_results.assignment_df = pd.read_pickle(
                    SETTINGS.annotations_dir / f"ass_{self.id}.pickle"
                )
                # this, cuz values are lists.
                # mp.raw_annotation_df['value'] = mp.raw_annotation_df['value'].apply(ast.literal_eval)
                # mp.raw_annotation_df['platform_id'] = mp.raw_annotation_df['platform_id'].astype(str)
                return ann_results
        # new file or there is no raw_dataframe
        # todo, assign within ann_results
        ann_results.raw_annotation_df, ann_results.assignment_df = (
            ann_results.get_annotation_df()
        )
        ann_results.raw_annotation_df.to_pickle(
            SETTINGS.annotations_dir / f"raw_{self.id}.pickle"
        )
        ann_results.assignment_df.to_pickle(
            SETTINGS.annotations_dir / f"ass_{self.id}.pickle"
        )
        self._ann_results = ann_results
        return ann_results

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

    def store_agreement_report(
        self, agreement_report: "Agreements", gen_csv_tables: bool = True
    ) -> list[Path]:
        raw_dest = self.path_for(SETTINGS.agreements_dir)
        paths = [raw_dest]
        raw_dest.write_text(
            json.dumps(
                {
                    var: res.model_dump(exclude_defaults=True)
                    for var, res in agreement_report.results.items()
                },
                indent=2,
            )
        )

        if gen_csv_tables:
            dest = self.path_for(SETTINGS.agreements_dir, ext=".csv")
            paths.append(dest)
            variables = self.variables()
            writer = DictWriter(
                dest.open("w", encoding="utf-8"),
                fieldnames=[
                    "variable",
                    "type",
                    "option",
                    "fleiss",
                    "gwet",
                    "ratio",
                    "abs",
                    "total",
                ],
            )
            writer.writeheader()

            for var_agreement in agreement_report.results.values():
                var = var_agreement.variable
                _choice_type = cast(ChoiceVariableModel, variables[var]).choice
                var_data = {"variable": var, "type": _choice_type}
                if _choice_type == "single":
                    row_data = var_data | {"option": "VARIABLE_LEVEL"}
                    for (
                        agreement_type,
                        agreement_value,
                    ) in var_agreement.single_overall.items():
                        row_data[agreement_type] = agreement_value or "NaN"
                    writer.writerow(row_data)

                for (
                    option,
                    option_agreements,
                ) in var_agreement.options_agreements.items():
                    row_data = var_data | {"option": option}
                    for (
                        agreement_type,
                        agreement_value,
                    ) in option_agreements.items():
                        row_data[agreement_type] = agreement_value
                    writer.writerow(row_data)

                    if _choice_type == "multiple":
                        row_data = var_data | {"option": f"{option}-SEL"}
                        for (
                            agreement_type,
                            agreement_value,
                        ) in var_agreement.multi_select_inclusion_agreement[
                            option
                        ].items():
                            row_data[agreement_type] = agreement_value
                        writer.writerow(row_data)

        conflicts_dest = self.path_for(SETTINGS.agreements_dir)
        conflicts_dest = (
            conflicts_dest.parent / f"{conflicts_dest.stem}_conflicts.json"
        )
        paths.append(conflicts_dest)
        save_json(conflicts_dest, agreement_report.option_tasks)

        return paths

    def get_agreement_data(self) -> dict[str, AgreementResult]:
        # todo test
        return {
            var: AgreementResult.model_validate(ag)
            for var, ag in json.load(
                self.path_for(SETTINGS.agreements_dir).open()
            ).items()
        }

    def get_tasks(self) -> LSTaskList:
        return LSTaskList.model_validate_json(
            self.path_for(SETTINGS.tasks_dir).read_text()
        )

    def save_tasks(
        self, tasks: list[LSTask], include_additional: set[str] = None
    ) -> Path:
        if not include_additional:
            include_additional = set()
        dest = self.path_for(SETTINGS.tasks_dir)
        dest.write_text(
            json.dumps(
                [
                    t.model_dump(
                        include={"data", "id", "project"} | include_additional
                    )
                    for t in tasks
                ],
                indent=2,
            )
        )
        logger.info(f"Save tasks to: {dest.as_posix()}")
        return dest

    def store_temp_tasks(self, tasks: LSTaskList[LSTask]) -> Path:
        dest = self.path_for(SETTINGS.temp_file_path)
        dest.write_text(
            json.dumps(
                [
                    t.model_dump(include={"data", "id", "project"})
                    for t in tasks.root
                ],
                indent=2,
            )
        )
        logger.info(f"Save tasks to: {dest.as_posix()}")
        return dest


class ProjectOverview(BaseModel):
    projects: dict[ProjectAccess, ProjectData]
    alias_map: dict[str, ProjectData] = Field(
        default_factory=dict, exclude=True
    )
    default_map: dict[PlLang, ProjectData] = Field(
        default_factory=dict, exclude=True
    )

    @model_validator(mode="after")
    def post_build(cls, overview: "ProjectOverview") -> "ProjectOverview":
        overview.create_map()
        return overview

    def create_map(self):
        """
        create alias_map and default_map
        """
        self.alias_map = {}
        self.default_map = {}
        for project in self.projects.values():
            # print(project.id, project.name)
            if project.alias in self.alias_map:
                print(f"warning: alias {project.alias} already exists")
                continue
            self.alias_map[project.alias] = project
            pl_l = (project.platform, project.language)

            # is the project has the default flag...
            if project.default:
                # check if the already set default, actually has the flat
                if set_default := self.default_map.get(pl_l, None):
                    if set_default.default:
                        print(
                            f"warning: default {pl_l} already exists. Not setting {project.title} as default"
                        )
                        continue
                self.default_map[pl_l] = project
            # just set the first pl_l into the default map
            elif pl_l not in self.default_map:
                self.default_map[pl_l] = project

    @staticmethod
    def load() -> "ProjectOverview":
        pp = SETTINGS.projects_main_file
        if not pp.exists():
            print("projects file missing")
        # print(pp.read_text())
        # print(ProjectOverView2.model_validate_json(pp.read_text()))
        return ProjectOverview.model_validate(
            {"projects": json.loads(pp.read_text())}
        )

    def get_project(self, p_access: ProjectAccess) -> ProjectData:
        # int | str | platf_lang_default | platform_lang_name
        if isinstance(p_access, int):
            return self.projects[str(p_access)]
        elif isinstance(p_access, str):
            return self.alias_map[p_access]
        elif (is_t := isinstance(p_access, tuple)) and (
            length := len(p_access)
        ) == 2:
            return self.default_map[p_access]
        elif is_t and length == 4:
            id, alias, platform, language = p_access
            if alias:
                return self.alias_map[alias]
            elif id:
                return self.projects[str(id)]
            elif platform and language:
                return self.default_map[(platform, language)]
            raise ValueError(
                f"\n\nYour project parameters where not right: {id=} {platform=} {language=}, {alias=} not a valid project-access"
            )
        raise ValueError(f"unknown project access: {p_access}")

    def create(
        self,
        p: ProjectCreate,
        add_coding_game_view: Optional[bool] = True,
        maximum_annotations: Optional[int] = 2,
    ) -> ProjectData:
        if p.alias in self.alias_map:
            raise ValueError(f"alias {p.alias} already exists")
        if p.default:
            if default_ := self.default_map[(p.platform, p.language)]:
                if default_.default:
                    raise ValueError(f"default {p.pl_lang} already exists")

        params = {
            "color": "#617ada",
            "maximum_annotations": maximum_annotations,
            "sampling": "Uniform sampling",
            "show_collab_predictions": True,
        }
        model = ProjectModel(
            title=p.title,
            description=p.full_description,
            **params,
        )
        project_model = ls_client().create_project(model)
        p_i = ProjectData(id=project_model.id, **p.model_dump())
        if add_coding_game_view:
            p_i.create_view(
                ProjectViewCreate.model_validate(
                    {"project": project_model.id, "data": {}}
                )
            )

        self.projects[p_i.id] = p_i
        if p_i.default:
            self.default_map[p.pl_lang] = p_i
        self.alias_map[p.alias] = p_i
        self.save()

        print(f"project created and saved: {repr(p_i)}")
        dest = p_i.path_for(SETTINGS.labeling_templates, ext=".xml")
        dest.write_text(project_model.label_config)

        return p_i

    def save(self):
        projects = {p.id: p for p in self.projects.values()}
        SETTINGS.projects_main_file.write_text(
            json.dumps(
                {id: p.model_dump() for id, p in projects.items()}, indent=2
            )
        )

    def project_list(self) -> list[ProjectData]:
        return list(self.projects.values())


platforms_overview: ProjectOverview = ProjectOverview.load()


def get_project(
    id: Optional[int] = None,
    alias: Optional[str] = None,
    platform: Optional[str] = None,
    language: Optional[str] = None,
) -> ProjectData:
    po = platforms_overview.get_project((id, alias, platform, language))
    logger.debug(repr(po))
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

    def clean_annotation_results(
        self, simplify_single: bool = True, variables: set[str] = None
    ) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
        """

        :param simplify_single:
        :param variables:
        :return: filepath and result-dict: platform_id: [{coder-results}]
        """
        logger.info("Building raw annotations dataframe")

        def var_method(k, fix):
            if fix.deprecated:
                return None
            if fix.name_fix:
                return fix.name_fix
            return k

        extension_keys = set(self.project_data.variable_extensions.extensions)
        po_variables = self.project_data.variables(False)

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
            # print(task.id)
            for ann in task.annotations:
                ann_result = {}
                # print(f"{task.id=} {ann.id=}")
                if ann.was_cancelled:
                    # print(f"{task.id=} {ann.id=} C")
                    continue
                # print(f"{task.id=} {ann.id=} {len(ann.result)=}")

                for q_id, question in enumerate(ann.result):
                    orig_name = question.from_name
                    if orig_name not in extension_keys:
                        raise ValueError(
                            f"{orig_name} Not found in extensions. Update extension of project {repr(self.project_data)}"
                        )
                    new_name = q_extens.get(question.from_name)

                    if not new_name:
                        print(f"unknown variable... {question.from_name}")
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
            alternative=f"clean_{self.project_data.id}",
        )
        dest.write_text(json.dumps(results))
        print(f"Clean results written to {dest}")
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
        logger.info("Building raw annotations dataframe")
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
                        print(f"unknown variable... {question.from_name}")
                        continue
                    var_def = variables[new_name]
                    idx = 0
                    if var_def.group_name:
                        new_name = var_def.group_name
                        idx = var_def.group_index
                    # print(question)
                    if question.type == "choices":
                        type_ = cast(
                            ChoiceVariableModel, variables[new_name]
                        ).choice
                    elif question.type == "textarea":
                        type_ = "text"
                    else:
                        print("unknown question type")
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

        """
        raw_annotation_df = df.astype(
            {"task_id": "int32", "ann_id": "int32", 'user_id': "category", "value_idx": "int32",
             'type': "category", 'question': "category", 'value': "string"})
        """
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
        logger.info(
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
