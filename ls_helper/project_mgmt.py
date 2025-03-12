from pathlib import Path

import orjson

from ls_helper.ana_res import parse_label_config_xml
from ls_helper.models import ProjectOverview, MyProject
from ls_helper.settings import SETTINGS


class ProjectMgmt:

    @staticmethod
    def update_projects():
        projects_data = SETTINGS.client.projects_list()
        project_map = {p.id: p for p in projects_data}

        projects_info = ProjectOverview.projects()
        for platform, p_data in projects_info:
            # print(platform, p_data)
            for lang, l_d in p_data.items():
                # print(lang, l_d)
                if _id := l_d.id:
                    if _id not in project_map:
                        print(
                            f"warning: project-info has {platform}.{lang} with id: {_id} but does not exist in LS data")
                    dest = projects_info.project_data_path(platform, lang)
                    ls_project_data = project_map[_id]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(
                        orjson.dumps(ls_project_data.model_dump(), option=orjson.OPT_INDENT_2).decode("utf-8"),
                        encoding="utf-8")

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
