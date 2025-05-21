import json
import re
from csv import DictWriter
from datetime import datetime, timedelta
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast, Annotated

import pandas as pd
from lxml.etree import ElementTree
from pydantic import BaseModel, ConfigDict, Field, model_validator, PrivateAttr

from ls_helper.agreements_calculation import AgreementResult
from ls_helper.build_configs import (
    LabelingInterfaceBuildConfig,
    build_from_template,
)
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.funcs import get_latest_annotation_file
from ls_helper.models.interface_models import (
    IChoices,
    InterfaceData,
    ProjectVariableExtensions,
)
from ls_helper.models.result_models import ProjectResult, ProjectAnnotationResultsModel
from ls_helper.models.variable_models import (
    VariableModel,
    ChoiceVariableModel,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectModel as LSProjectModel,
    ProjectModel,
    ProjectViewCreate,
    ProjectViewModel,
    TaskResultModel,
    Task as LSTask,
    TaskList as LSTaskList,
    TaskCreateList as LSTaskCreateList
)
from ls_helper.settings import (
    SETTINGS,
    ls_logger,
    TIMESTAMP_FORMAT,
)
from tools.files import save_json, read_data
from tools.project_logging import get_model_logger

if TYPE_CHECKING:
    from ls_helper.agreements_calculation import Agreements

PlLang = tuple[str, str]
ProjectAccess = (
        int
        | str
        | PlLang
        | tuple[Optional[int], Optional[str], Optional[str], Optional[str]]
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


class ProjectTasks:

    def __init__(self, project_data: "ProjectData") -> None:
        self._pd = project_data

    @property
    def fp(self):
        return self._pd.path_for(SETTINGS.tasks_dir)

    def download_tasks(self, save: bool = True) -> LSTaskList:
        tasks = LSTaskList.model_validate(ls_client().get_task_list(project=self._pd.id))
        if save:
            self.save(tasks)
        return tasks

    def get(self, download_when_missing: bool = True, download: bool = False) -> LSTaskList:
        missing = not self.fp.exists()
        if download or (missing and download_when_missing):
            tasks = self.download_tasks(save=True)
            return tasks
        if missing:
            raise FileNotFoundError(f"No tasks file found: {self.fp}")
        return LSTaskList.model_validate_json(
            self.fp.read_text()
        )

    def import_tasks(self, tasks: LSTaskCreateList) -> LSTaskList:
        tasks = ls_client().import_tasks(self._pd.id, tasks)
        self.save(tasks)
        return tasks

    def save(self, tasks: LSTaskList, include_additional_fields: Optional[set[str]] = None) -> Path:
        if not include_additional_fields:
            include_additional_fields = set()
        self.fp.write_text(
            json.dumps(
                [
                    t.model_dump(
                        include={"data", "id", "project"} | include_additional_fields
                    )
                    for t in tasks.root
                ],
                indent=2,
            )
        )
        self._pd._logger.info(f"Save tasks to: {self.fp.as_posix()}")
        return self.fp


class ProjectViews:

    def __init__(self, project_data: "ProjectData") -> None:
        self._pd = project_data

    def download(self):
        views = ls_client().get_project_views(self.id)
        self._pd.path_for(SETTINGS.view_dir).write_text(
            json.dumps([v.model_dump() for v in views], indent=2)
        )
        self._pd._logger.info(f"refresh views: {views}")
        return views

    def delete(self, view_id: int) -> bool:
        return ls_client().delete_view(view_id)

    def get(self) -> list[ProjectViewModel]:
        view_file = self._pd.path_for(SETTINGS.view_dir)
        if not view_file.exists():
            self.download()
        data = json.load(view_file.open())
        return [ProjectViewModel.model_validate(v) for v in data]

    def create(self, create_data: ProjectViewCreate) -> ProjectViewModel:
        if not create_data.data.hiddenColumns:
            create_data.data.hiddenColumns = read_data(
                SETTINGS.BASE_DATA_DIR
                / "default/ls_project_view_hiddenColumns.json"
            )
        return ls_client().create_view(create_data)


class ProjectData(ProjectCreate):
    id: int
    _project_data: Optional[LSProjectModel] = None
    _interface_data: Optional[InterfaceData] = None
    _variable_extensions: Optional[ProjectVariableExtensions] = None
    _ann_results: Optional["ProjectResult"] = None
    _logger: Optional[Logger] = None
    _tasks: ProjectTasks = PrivateAttr()
    _view: ProjectViews = PrivateAttr()

    model_config = ConfigDict()

    def model_post_init(self, context: Any) -> None:
        self._logger = get_model_logger(self)
        self._tasks = ProjectTasks(self)
        self._view = ProjectViews(self)

    @property
    def tasks(self) -> ProjectTasks:
        return self._tasks

    @property
    def views(self) -> ProjectViews:
        return self._view

    def path_for(
            self,
            base_p: Path,
            alternative: Optional[str] = None,
            ext: Optional[str] = ".json",
    ) -> Path:
        if not ext:
            ext = ".json"
        # todo. temp info
        if alternative:
            self._logger.info(f"Note, that the project-id does not need to be passed anymore for 'alternative'")
        name = f"{alternative}_{self.id}" if alternative else self.id
        return base_p / f"{name}{ext}"

    def __repr__(self) -> str:
        return f"{self.id}/{self.alias}/{self.platform}/{self.language}"

    @property
    def project_data(self) -> LSProjectModel:
        if self._project_data:
            return self._project_data

        def guarantee_ls_project_data():
            if not self.path_for(SETTINGS.projects_dir).exists():
                project_data = ls_client().get_project(self.id)
                if not project_data:
                    raise ValueError(f"No project found: {self.id}")
                self.save_project_data(project_data)

        guarantee_ls_project_data()
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
        self._logger.info(f"project-data saved for {repr(self)}: -> {dest}")

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
        self._logger.info(
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
        self._logger.info(f"Save {type(data).__name__} to: {p}")

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

    def get_recent_annotations(
            self,
            accepted_age: int = 6,
            use_existing: bool = False,
    ) -> Optional[
        tuple[
            Annotated[bool, "use_local"],
            Optional["ProjectAnnotationResultsModel"],
        ]
    ]:
        """
        todo. unify it with (get_annotations_results)
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
                self._logger.info(
                    f"Get recent, gets latest annotation: {file_dt:%m%d_%H%M}"
                )

                annotation_file = get_latest_annotation_file(self.id)
                if not annotation_file:
                    self._logger.warning("No annotation file?!")
                    return None
                task_results = [
                    TaskResultModel.model_validate(t)
                    for t in json.load(annotation_file.open(encoding="utf-8"))
                ]

                return True, ProjectAnnotationResultsModel(
                    task_results=task_results, timestamp=file_dt
                )

        # todo this stuff is old. needs refactoring love. move to ProjectData model
        self._logger.info("downloading annotations")
        result = ls_client().get_project_annotations(self.id)
        if not result:
            return False, None
        ts = datetime.now()
        base_path = SETTINGS.annotations_dir / str(self.id)
        res_path = (
                base_path
                / f"{ts.strftime(TIMESTAMP_FORMAT)}.json"
        )

        res_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"dumping project annotations to {res_path}")
        pa = ProjectAnnotationResultsModel(task_results=result, timestamp=ts)
        json.dump(
            [r.model_dump() for r in result],
            res_path.open("w", encoding="utf-8"),
        )
        # remove old annotation files:
        all_files = list(sorted(base_path.glob("*.json"), reverse=True))
        for old_files in all_files[SETTINGS.ANNOTATIONS_HISTORY_LENGTH:]:
            self._logger.debug(f"Removing old annotation file: {old_files}")
            old_files.unlink()
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
                self._logger.info(f"variable from extensions is redundant {var}")
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
        get the raw annotation results (in form of the two fundamental dataframes)
        will store those dataframes for a project as 'raw_<id>' and 'ass_<id>' pickle files.
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
        self._logger.info(f"Save tasks to: {dest.as_posix()}")
        return dest


class ProjectOverview(BaseModel):
    projects: dict[ProjectAccess, ProjectData]
    alias_map: dict[str, ProjectData] = Field(
        default_factory=dict, exclude=True
    )
    default_map: dict[PlLang, ProjectData] = Field(
        default_factory=dict, exclude=True
    )

    def model_post_init(self, context: Any, /) -> None:
        self._logger = get_model_logger(self)

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
                self._logger.warning(f"alias {project.alias} already exists")
                continue
            self.alias_map[project.alias] = project
            pl_l = (project.platform, project.language)

            # is the project has the default flag...
            if project.default:
                # check if the already set default, actually has the flat
                if set_default := self.default_map.get(pl_l, None):
                    if set_default.default:
                        self._logger.warning(
                            f"default {pl_l} already exists. Not setting {project.title} as default"
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
            ls_logger.warning("projects file missing")
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
        """
        TODO coding game view is not created ?!?!
        :param p:
        :param add_coding_game_view:
        :param maximum_annotations:
        :return:
        """
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
            p_i.views.create(
                ProjectViewCreate.model_validate(
                    {"project": project_model.id, "data": {}}
                )
            )

        self.projects[p_i.id] = p_i
        if p_i.default:
            self.default_map[p.pl_lang] = p_i
        self.alias_map[p.alias] = p_i
        self.save()

        self._logger.info(f"project created and saved: {repr(p_i)}")
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
    ls_logger.debug(repr(po))
    return po


ProjectResult.model_rebuild()
