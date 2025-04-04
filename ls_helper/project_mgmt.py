import json
from typing import Any, Optional

import orjson

from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import ProjectModel, ProjectViewModel, ProjectViewCreate
from ls_helper.new_models import ProjectCreate, platforms_overview2, ProjectInfo
from ls_helper.settings import SETTINGS
from tools.env_root import root
from tools.files import read_data


class ProjectMgmt:
    DEFAULT_VIEW_HIDDEN_COLUMNS_FP = root() / "data/ls_data/default/ls_project_view_hiddenColumns.json"

    @staticmethod
    def default_project_values() -> dict[str, Any]:
        return {
            "color": "#617ada",
            "maximum_annotations": 2,
            "sampling": "Uniform sampling"
        }


    @staticmethod
    def update_projects():
        """
        TODO, not sure if still used or required.
        :return:
        """
        projects_data = SETTINGS.client.projects_list()
        project_map = {p.id: p for p in projects_data}

        projects_info = platforms_overview2
        for platform, p_data in projects_info:
            # print(platform, p_data)
            for lang, l_d in p_data.items():
                # print(lang, l_d)
                if _id := l_d.id:
                    if _id not in project_map:
                        print(
                            f"warning: project-info has {platform}.{lang} with id: {_id} but does not exist in LS data")
                    dest = p_data.project_data_path(platform, lang)
                    ls_project_data = project_map[_id]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(
                        orjson.dumps(ls_project_data.model_dump(), option=orjson.OPT_INDENT_2).decode("utf-8"),
                        encoding="utf-8")

    @classmethod
    def create_view(cls, view: ProjectViewCreate) -> ProjectViewModel:
        if not view.data.hiddenColumns:
            view.data.hiddenColumns = read_data(cls.DEFAULT_VIEW_HIDDEN_COLUMNS_FP)
        return ls_client().create_view(view)

    @classmethod
    def refresh_views(self, po: ProjectInfo):
        views = ls_client().get_project_views(po.id)
        po.view_file.write_text(json.dumps([v.model_dump() for v in views]))



    @classmethod
    def create_project(cls, p: ProjectCreate, add_coding_game_view: bool = True) -> tuple[
        ProjectModel, Optional[ProjectViewModel]]:
        model = ProjectModel(title=p.title,
                             description=p.full_description,
                             **ProjectMgmt.default_project_values())
        project = ls_client().create_project(model)
        coding_game_view: Optional[ProjectViewModel] = None
        if add_coding_game_view:
            coding_game_view = cls.create_view(ProjectViewCreate.model_validate(
                {"project": project.id, "data": {}}))

        return project, coding_game_view

    # @staticmethod
    # def get_annotations(platform: str, language: str):
    #     project_data = ProjectOverview.project_data(platform, language)
    #     project_id = project_data["id"]
    #
    #     conf = parse_label_config_xml(project_data["label_config"],
    #                                   project_id=project_id,
    #                                   include_text=True)
    #
    #     annotations = SETTINGS.client.get_project_annotations(project_id)
    #
    #     mp = MyProject(project_data=project_data, annotation_structure=conf,
    #                    raw_annotation_result=annotations)
    #     mp.calculate_results()
    #     mp.results2csv(Path("t.csv"), with_defaults=False)
    #     return mp
