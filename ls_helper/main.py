import json
from pathlib import Path

from ana_res import parse_label_config_xml, get_config_project_project_data
from ls_helper.funcs import get_latest_annotation, test_update_other_coding_game, update_coding_game
from ls_helper.models import MyProject, ProjectAnnotationExtension
from my_labelstudio_client.client import LabelStudioBase
from settings import SETTINGS

if __name__ == "__main__":
    client = LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)

    # all_projects = client.projects_list()
    # for p in all_projects:
    #     print(p.id, p.title)

    # project info
    # res = client.get_project(29)
    # json.dump(json.loads(orjson.dumps(res.model_dump())),open("data/project.json","w"), indent=2)

    # patch project
    res = client.patch_project(33, {
        "maximum_annotations": 2,
        # "sampling": "Uniform sampling"
    })


    p_id = 33 # real yt en
    # download annotations and update the others,coding game data
    project_annotations = client.get_project_annotations(33)
    others, coding_game = test_update_other_coding_game(project_annotations, p_id)
    update_coding_game(client, p_id, True,56, platform_ids=coding_game)
    exit()
    # download_project_views(29)

    project_annotations = get_latest_annotation(29)

    """
    MyProject.annotation_result: ProjectAnnotationResults
    ProjectAnnotationResults.annotation_results: list[TaskAnnotResults]
    TaskAnnotResults.items: Optional[dict[str, TaskAnnotationItem]] = Field(default_factory=dict)
    
    get_latest_annotation -> ProjectAnnotations
    ProjectAnnotations.annotations: list["TaskResultModel"]
    TaskResultModel.annotations: list[TaskAnnotationModel]
    TaskAnnotationModel.result: list[AnnotationResult]
    """

    client.get_users()
    # create annotation, agreement table
    project_data = json.load(open("data/project_29.json", encoding="utf-8"))
    # project_views = client.get_project_views(29)
    project_views = None
    conf = parse_label_config_xml(get_config_project_project_data(project_data),
                                  project_id=29,
                                  include_text=True, include_text_names=["source", "p_id_v"])

    mp = MyProject(project_data=project_data,
                   annotation_structure=conf,
                   raw_annotation_result=project_annotations,
                   project_views=project_views,
                   data_extensions=
                   ProjectAnnotationExtension.model_validate(json.load(open(f"data/fixes/{p_id}.json"))))

    mp.calculate_results()

    # mp.apply_extension(False)
    # mp.results2csv(Path("t-raw.csv"))
    mp.apply_extension(True)
    mp.results2csv(Path("t.csv"), with_defaults=False)
    # ano_d = [TaskResultModel.model_validate(_) for _ in project_annotations]

    # hot fix: better names for inputs, json file generation
    # name_fix = q1_twitter_yt.create_proper_names_titles(conf)
    # ng = json.dumps(name_fix.model_dump())
    # mp.data_extensions = name_fix

    #
    # apply_fixes(project_annotations, name_fix)

    # CSV
    # mp.annotation_result = results2csv(conf, project_annotations, , mp.data_extensions)
    # upload_disagreement_selection(mp.annotation_result, 38)
    #

    pass
    # print(conf)
