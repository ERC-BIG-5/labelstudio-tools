import csv
import json
from pathlib import Path
from typing import Optional

import httpx
import jsonpath_ng

from ls_helper.models import VariableExtension, ResultStruct, ProjectAnnotationExtension, ProjectAnnotations, MyProject
from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import UserModel
from settings import SETTINGS


def test_update_other_coding_game(project_annotations: ProjectAnnotations) -> tuple[dict[str, list[str]], list[str]]:
    """
    pass in the results
    Parameters
    ----------
    results

    Returns
    -------

    """
    csv_file = Path("data/info/annotations.csv")
    coding_game_file = Path("data/info/coding_game.json")

    header = ['Nature Element/Process', 'Human-Nature Action']

    others = {h: [] for h in header}

    for_coding_game = []

    for task_res in project_annotations.annotations:
        annotations = task_res.annotations
        for annotation in annotations:
            for result in annotation.result:
                if result.from_name == "ne_o_t_0_t":
                    val = result.value.text[0]
                    if val not in others[header[0]]:
                        others[header[0]].append(val)
                elif result.from_name == "sa_o_t_0_t":
                    val = result.value.text[0]
                    if val not in others[header[1]]:
                        others[header[1]].append(val)

                elif result.from_name == "for_coding_game":
                    if result.value.choices[0] == "Yes":
                        p_id = task_res.data["platform_id"]
                        if p_id not in for_coding_game:
                            for_coding_game.append(p_id)

    ne_list = others[header[0]]
    sa_list = others[header[1]]
    max_len = max(len(ne_list), len(sa_list))

    # Write back to CSV using DictWriter
    with open(csv_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for i in range(max_len):
            row = {
                header[0]: ne_list[i] if i < len(ne_list) else '',
                header[1]: sa_list[i] if i < len(sa_list) else ''
            }
            writer.writerow(row)

    json.dump(for_coding_game, open(coding_game_file, "w", newline='', encoding='utf-8'))

    print(others)
    print(for_coding_game)
    return others, for_coding_game


def pick_and_flatten(results):
    """
    not rally usefull...
    Parameters
    ----------
    results

    Returns
    -------

    """
    all_keys: dict[str, list] = {}
    task_results: list[dict[str, list]] = []
    for t_idx, task in enumerate(results):
        task_res = {"p_id": task["data"]["platform_id"], "platform": task["data"]["source"
                                                                                  ""]}
        task_results.append(task_res)

        for an_idx, task_annot in enumerate(task['annotations']):
            # f_annot = flatten(task_annot,reducer=make_reducer(delimiter='.'),enumerate_types=(list,))
            # print(f_annot)
            for res in task_annot['result']:
                values = jsonpath_ng.parse("$.*").find(res['value'])
                def_val = values[0].value
                all_keys.setdefault(res["from_name"], []).append(def_val)
                task_res.setdefault(res["from_name"], []).append(def_val)
            # name_path = jsonpath_ng.parse("$.result[*].from_name")
            # match = [match.value for match in name_path.find(task_annot)]
            # print(t_idx, an_idx, match)
    # print(json.dumps(all_keys, indent=2))
    print(json.dumps(task_results, indent=2))
    pass


def update_coding_game(project_id: int, view_id: int, platform_ids: list[str] = ()):
    # TODO: how is this working, project_id should be a query param
    resp = httpx.get(f"{SETTINGS.LS_HOSTNAME}/api/dm/views/project={project_id}", headers={
        "Authorization": f"Token {SETTINGS.LS_API_KEY}"
    })

    current = resp.json()
    print(current)

    new_items = []
    new_filters = {"conjunction": "or", "items": new_items}
    current["data"]["filters"] = new_filters
    for p_id in platform_ids:
        # print(for_coding_game)
        new_items.append({
            "filter": "filter:tasks:data.platform_id",
            "operator": "equal",
            "type": "String",
            "value": p_id
        })

    res = {"data": {"title": "Coding game", "filters": new_filters}}
    print(res)
    resp = httpx.patch(f"{SETTINGS.LS_HOSTNAME}/api/dm/views/{project_id}", headers={
        "Authorization": f"Token {SETTINGS.LS_API_KEY}"
    }, json=res)
    if resp.status_code != 200:
        print(f"error updating view for coding game: {resp.status_code}")


def get_latest_annotation(project_id: int) -> Optional[ProjectAnnotations]:
    base_dir = Path(f"data/annotations/")
    annotation_files = list((base_dir / str(project_id)).glob("*.json"))
    if not annotation_files:
        print("no annotations found")
        return None
    annotation_file = sorted(annotation_files)[-1]
    return ProjectAnnotations(project_id=project_id,
                              annotations=json.load(annotation_file.open(encoding="utf-8")),
                              file_path=annotation_file)


def generate_result_fixes_template(annotation_struct: ResultStruct) -> ProjectAnnotationExtension:
    data: dict[str, VariableExtension] = {}

    for field in annotation_struct.inputs:
        data[field] = VariableExtension()

    return ProjectAnnotations()


def apply_fixes(annotations: ProjectAnnotations, fixes: ProjectAnnotationExtension):
    pass


def download_project_views(project_id: int, store: bool = True):
    views_resp = httpx.get(f"{SETTINGS.LS_HOSTNAME}/api/dm/views/?project={project_id}", headers={
        "Authorization": f"Token {SETTINGS.LS_API_KEY}"
    })
    if views_resp.status_code == 200:
        data = views_resp.json()

    if store:
        view_dir = SETTINGS.BASE_DATA_DIR / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(f"{view_dir}/{project_id}.json", "w", encoding="utf-8"))
    return data


def update_project_view(p: MyProject,
                        view_id: Optional[int],
                        view_name: Optional[str] = None):
    if not p.project_views:
        resp = httpx.get(f"{SETTINGS.LS_HOSTNAME}/api/dm/views/project={p.project_id}", headers={
            "Authorization": f"Token {SETTINGS.LS_API_KEY}"
        })
        # raise NotImplemented("get the view through api")

def update_user_nicknames(refresh_users: bool = True):
    if refresh_users:
        client = LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)
        users = client.get_users()
    else:
        users = list(map(UserModel.model_validate, json.load(open("data/users.json", encoding="utf-8"))))
    nicknames_file = Path("data/user_nicknames.json")
    nicknames = json.load(nicknames_file.open(encoding="utf-8"))
    for idx, u in enumerate(users):
        print(u.model_dump(include={"id","first_name","last_name","username","initials"}))
        if str(u.id) not in nicknames:
            nicknames[u.id] = input("Nickname: ")

    nicknames_file.write_text(json.dumps(nicknames, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    data = json.load(open("data/annotations/29-20250210_1402.json"))
    # test_update_other_coding_game()
    pick_and_flatten(data)
