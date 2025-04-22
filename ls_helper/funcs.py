import csv
import json
from pathlib import Path
from typing import Optional, Literal

import httpx
import jsonpath_ng

from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import (
    UserModel,
    ProjectViewModel,
    TaskResultModel,
)
from ls_helper.settings import SETTINGS


def test_update_other_coding_game(
    annotations: list[TaskResultModel], project_id: int
) -> tuple[dict[str, list[str]], list[str]]:
    """
    TODO THIS FUNC NEEDS TO GO
    pass in the results
    Parameters
    ----------
    results

    Returns
    -------

    """
    csv_file = Path(f"data/info/annotations_{project_id}.csv")
    coding_game_file = Path(f"data/info/coding_game_{project_id}.json")

    header = ["Nature Element/Process", "Human-Nature Action"]

    others = {h: [] for h in header}

    for_coding_game = []
    data_for_coding_game = []

    for task_res in annotations:
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
                            data_for_coding_game.append(task_res.data)

    ne_list = others[header[0]]
    sa_list = others[header[1]]
    max_len = max(len(ne_list), len(sa_list))

    # Write back to CSV using DictWriter
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for i in range(max_len):
            row = {
                header[0]: ne_list[i] if i < len(ne_list) else "",
                header[1]: sa_list[i] if i < len(sa_list) else "",
            }
            writer.writerow(row)

    print(f"coding game: ->  {coding_game_file}")
    json.dump(
        for_coding_game,
        open(coding_game_file, "w", newline="", encoding="utf-8"),
    )

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
        task_res = {
            "p_id": task["data"]["platform_id"],
            "platform": task["data"]["source"],
        }
        task_results.append(task_res)

        for an_idx, task_annot in enumerate(task["annotations"]):
            # f_annot = flatten(task_annot,reducer=make_reducer(delimiter='.'),enumerate_types=(list,))
            # print(f_annot)
            for res in task_annot["result"]:
                values = jsonpath_ng.parse("$.*").find(res["value"])
                def_val = values[0].value
                all_keys.setdefault(res["from_name"], []).append(def_val)
                task_res.setdefault(res["from_name"], []).append(def_val)
            # name_path = jsonpath_ng.parse("$.result[*].from_name")
            # match = [match.value for match in name_path.find(task_annot)]
            # print(t_idx, an_idx, match)
    # print(json.dumps(all_keys, indent=2))
    print(json.dumps(task_results, indent=2))
    pass


def update_coding_game(
    client: LabelStudioBase,
    project_id: int,
    use_stored_data_if_available: bool,
    view_id: int,
    platform_ids: list[str] = (),
):
    # TODO @deprecated, use func below
    viwes = client.get_project_views(project_id)

    print(viwes)
    view_data = [c for c in viwes if c.id == view_id]
    if not view_data:
        print("no such view")
        return
    view_data = view_data[0]
    new_items = []
    new_filters = {"conjunction": "or", "items": new_items}
    view_data.data.filters = new_filters
    for p_id in platform_ids:
        # print(for_coding_game)
        new_items.append(
            {
                "filter": "filter:tasks:data.platform_id",
                "operator": "equal",
                "type": "String",
                "value": p_id,
            }
        )

    res = {
        "data": {
            "title": "Coding game",
            "filters": new_filters,
            "hiddenColumns": view_data.data.hiddenColumns,
        }
    }
    print(res)
    resp = client.patch_view(view_id, res)
    # resp = httpx.patch(f"{SETTINGS.LS_HOSTNAME}/api/dm/views/{view_id}", headers={
    #     "Authorization": f"Token {SETTINGS.LS_API_KEY}"
    # }, json=res)
    if resp.status_code != 200:
        print(f"error updating view for coding game: {resp.status_code}")
        print(resp.json())


def build_platform_id_filter(
    platform_ids: list[str | int],
    ls_main_field: Literal["platform_id", "task_id"],
):
    """
    should be simpler. this is only that complicated, cuz of the bad conflict models.
    :param platform_ids:
    :param ls_main_field:
    :return:
    """
    new_items = []
    new_filters = {"conjunction": "or", "items": new_items}
    filter_term = (
        "filter:tasks:data.platform_id"
        if ls_main_field == "platform_id"
        else "filter:tasks:id"
    )
    filter_type = "String" if ls_main_field == "platform_id" else "Number"
    for p_id in platform_ids:
        # print(for_coding_game)
        new_items.append(
            {
                "filter": filter_term,
                "operator": "equal",
                "type": filter_type,
                "value": p_id,
            }
        )
    return new_filters


def build_view_with_filter_p_ids(
    client: LabelStudioBase, view: ProjectViewModel, platform_ids: list[str]
):
    new_filters = build_platform_id_filter(platform_ids)

    res = {
        "data": {
            "title": view.data.title,
            "filters": new_filters,
            "hiddenColumns": view.data.hiddenColumns,
        }
    }

    # print(res)
    resp = client.patch_view(view.id, res)
    if resp.status_code != 200:
        print(f"error updating view for coding game: {resp.status_code}")
        print(resp.json())


def get_latest_annotation_file(project_id: int) -> Optional[Path]:
    base_dir = SETTINGS.annotations_dir
    annotation_files = list((base_dir / str(project_id)).glob("*.json"))
    if not annotation_files:
        print("no annotations found")
        return None
    return sorted(annotation_files)[-1]


def download_project_views(project_id: int, store: bool = True):
    views_resp = httpx.get(
        f"{SETTINGS.LS_HOSTNAME}/api/dm/views/?project={project_id}",
        headers={"Authorization": f"Token {SETTINGS.LS_API_KEY}"},
    )
    if views_resp.status_code == 200:
        data = views_resp.json()

    if store:
        view_dir = SETTINGS.BASE_DATA_DIR / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        json.dump(
            data, open(f"{view_dir}/{project_id}.json", "w", encoding="utf-8")
        )
    return data


def update_user_nicknames(refresh_users: bool = True):
    if refresh_users:
        client = LabelStudioBase(
            base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY
        )
        users = client.get_users()
    else:
        users = list(
            map(
                UserModel.model_validate,
                json.load(open("data/users.json", encoding="utf-8")),
            )
        )
    nicknames_file = Path("data/user_nicknames.json")
    nicknames = json.load(nicknames_file.open(encoding="utf-8"))
    for idx, u in enumerate(users):
        print(
            u.model_dump(
                include={
                    "id",
                    "first_name",
                    "last_name",
                    "username",
                    "initials",
                }
            )
        )
        if str(u.id) not in nicknames:
            nicknames[u.id] = input("Nickname: ")

    nicknames_file.write_text(
        json.dumps(nicknames, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    data = json.load(open("data/annotations/29-20250210_1402.json"))
    # test_update_other_coding_game()
    pick_and_flatten(data)
