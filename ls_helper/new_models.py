from pathlib import Path
from typing import Optional

from pydantic import BaseModel, RootModel, Field, field_validator, model_validator

from ls_helper.models import ProjectAccess
from ls_helper.settings import SETTINGS


class ProjectInfo(BaseModel):
    id: int
    platform: str
    language: str
    name: str
    alias: Optional[str] = None
    default: Optional[bool] = False
    #
    label_config_template: Optional[str] = None
    label_config_additions: Optional[list[str]] = Field(default_factory=list)
    coding_game_view_id: Optional[int] = None

    @model_validator(mode="after")
    def post_build(cls, data: "ProjectInfo") -> None:
        if not data.alias:
            data.alias = data.name.lower().replace(" ", "_")

        return data

class ProjectOverView2(BaseModel):
    projects: dict[ProjectAccess, ProjectInfo]
    alias_map: dict[str, ProjectInfo] = Field(default_factory=dict, exclude=True)
    default_map: dict[tuple[str,str], ProjectInfo] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def create_map(cls, overview: "ProjectOverView2") -> "ProjectOverView2":
        for project in overview.projects.values():
            if project.alias in overview.alias_map:
                print(f"warning: alias {project.alias} already exists")
                continue
            overview.alias_map[project.alias] = project
            pl_l = (project.platform, project.language)
            # just set the first pl_l into the default map
            if pl_l not in overview.default_map:
                overview.default_map[pl_l] = project
            # is the project has the default flag...
            if project.default:
                # check if the already set default, actually has the flat
                if set_default := overview.default_map.get(pl_l, None):
                    if set_default.default:
                        print(f"warning: default {pl_l} already exists. Not setting {project.name} as default")
                        continue
                overview.default_map[pl_l] = project

    @staticmethod
    def load() -> "ProjectOverview2":
        pp = Path(SETTINGS.BASE_DATA_DIR / "projects2.json")
        return ProjectOverView2.model_validate_json(pp.read_text())

    def get_project(self, p_access: ProjectAccess) -> ProjectInfo:
        # int | str | platf_lang_default | platform_lang_name
        if isinstance(p_access, int):
            return self.projects[p_access]
        elif isinstance(p_access, str):
            return self.alias_map[p_access]
        elif isinstance(p_access, tuple):
            if len(p_access) == 2:
                assert isinstance(p_access[1], tuple) and len(p_access[1]) == 2
                return self.default_map[p_access]


platforms_overview2 = ProjectOverView2.load()
