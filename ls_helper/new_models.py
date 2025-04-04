import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ls_helper.ana_res import parse_label_config_xml
from ls_helper.models import ProjectAnnotationExtension, ResultStruct
from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel
from ls_helper.settings import SETTINGS
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
    def post_build(cls, data: "ProjectInfo") -> "ProjectInfo":
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


class ProjectInfo(ProjectCreate):
    id: int
    _project_data: Optional[ProjectModel] = None
    _annot_structure: Optional[ResultStruct] = None
    _annot_extension: Optional[ProjectAnnotationExtension] = None

    @property
    def project_data(self) -> ProjectModel:
        if self._project_data:
            return self._project_data
        fin: Optional[Path] = None
        if (p_p_l := SETTINGS.projects_dir / f"{self.platform}/{self.language}.json").exists():
            print(f"project data file for {self.id}: platform-language... change in future")
            fin = p_p_l
        elif (p_i := SETTINGS.projects_dir / f"{self.id}.json").exists():
            fin = p_i
        if not fin:
            raise FileNotFoundError(f"project data file for {self.id}: platform-language does not exist")
        self._project_data = ProjectModel.model_validate_json(fin.read_text())
        return self._project_data

    def get_structure(self,
                      include_text: bool = True,
                      apply_extension: bool = True) -> ResultStruct:
        """
        caches the structure.
        :param include_text:
        :param apply_extension:
        :return:
        """
        if self._annot_structure and apply_extension == self._annot_structure._extension_applied:
            return self._annot_structure

        self._annot_structure = parse_label_config_xml(self.project_data.label_config, include_text=include_text)
        if apply_extension:
            self._annot_structure.apply_extension(self.data_extension)
        return self._annot_structure

    @property
    def data_extension(self) -> ProjectAnnotationExtension:
        if self._annot_extension:
            return self._annot_extension
        if (fi := SETTINGS.fixes_dir / "unifixes.json").exists():
            data_extensions = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
        else:
            print(f"no unifixes.json file yet in {SETTINGS.fixes_dir / 'unifix.json'}")
            data_extensions = {}
        if (p_fixes_file := SETTINGS.fixes_dir / f"{self.id}.json").exists():
            p_fixes = ProjectAnnotationExtension.model_validate_json(p_fixes_file.read_text(encoding="utf-8"))
            data_extensions.fixes.update(p_fixes.fixes)
            data_extensions.fix_reverse_map.update(p_fixes.fix_reverse_map)
        self._annot_extension = data_extensions
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

    def check_fixes(self):
        """
        go through all fixes and mark those, which are not in the structure:
        :return:
        """
        structure = self.get_structure()
        for var in self.data_extension.fixes:
            if var not in structure:
                logger.warning(f"variable from fixes is redundant {var}")


class ProjectOverView2(BaseModel):
    projects: dict[ProjectAccess, ProjectInfo]
    alias_map: dict[str, ProjectInfo] = Field(default_factory=dict, exclude=True)
    default_map: dict[PlLang, ProjectInfo] = Field(default_factory=dict, exclude=True)

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

    def get_project(self, p_access: ProjectAccess) -> ProjectInfo:
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

        p_i = ProjectInfo(id=project_model.id, **p.model_dump())

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
