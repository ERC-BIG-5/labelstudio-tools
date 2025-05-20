from csv import DictWriter

from ls_helper.command import annotations
from ls_helper.command.view import download_project_views
from ls_helper.funcs import build_view_with_filter_p_ids
from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS


def depr_create_views():
    # project 50 (youtube-en) is not completed yet
    writer: dict[str, DictWriter] = {}
    vars = ["Food"]
    for d in vars:
        writer[d] = DictWriter(
            (SETTINGS.temp_file_path / f"extra_{d}.csv").open("w"),
            fieldnames=["project", "platform_id"],
        )
        writer[d].writeheader()

    for i in [43]:  # , 51]:  # [50]:  # [43, 50, 51]:  # tw-en,yt-en, tw-es,
        po = get_project(i)
        f, res = annotations.clean_results(i, variables={"extras"})
        # print(f)
        p_ids = []
        for p_id, coder_res in res.items():
            found_extra = []
            for res in coder_res:
                for d in vars:
                    if d in found_extra:
                        continue
                    if d in res.get("extras", []):
                        writer[d].writerow(
                            {
                                "project": f"{po.platform}/{po.language}",
                                "platform_id": p_id,
                            }
                        )
                        p_ids.append(p_id)
                        found_extra.append(d)

        new_project = {
            43: 53
        }
        # view = po.create_view(ProjectViewCreate.model_validate({"project": new_project[po.id],
        #                                                         "data": {"title": "food"}}))

        new_po = get_project(new_project[po.id])
        download_project_views(new_po.id)

        view_id = {
            43: 195
        }
        food_view = [v for v in new_po.view.get() if v.id == view_id.get(po.id)]
        if not food_view:
            print(f"No view found for {po.alias}")
            continue
        else:
            view = food_view[0]
        build_view_with_filter_p_ids(view, [p_id for p_id in p_ids])


def flat_food_mask(extras_col) -> bool:
    return any(["Food" in res for res in extras_col])


def filter_food(p_id):
    """
    get all food relevant posts in the flattened format
    :param p_id:
    :return:
    """
    po = get_project(p_id)
    flat = po.get_annotations_results().flatten_annotation_results()
    mask = flat['extras'].apply(flat_food_mask)
    flat_filtered = flat[mask]

    # too bad. more R friedly formats didnt work. writers complained...
    # import pyarrow as pa
    # import pyarrow.feather as feather
    # feather.write_feather(flat_filtered, f'food_{p_id}.feather')
    # import pyreadr
    # print(type(flat_filtered))
    # import pandas as pd
    # fresh_df = pd.DataFrame(flat_filtered)
    # pyreadr.write_rds(fresh_df, f"food_{p_id}.rds")

    flat_filtered.to_csv(f'food_{p_id}.csv')


if __name__ == "__main__":
    # 53, twitter-en-5
    # 54 twitter-es-5
    filter_food(54)
