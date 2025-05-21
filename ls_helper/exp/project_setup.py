"""
guideline for setting up a project (example Weibo)
"""
import json
from csv import DictReader
from pathlib import Path

from ls_helper.models.main_models import get_project
from ls_helper.my_labelstudio_client.models import TaskCreateList as LSTaskCreateList


def _create_project():
    from ls_helper.command.project_setup import create_project
    create_project("Weibo p.v5 (1)", "weibo-ch-5", "weibo", "ch", 2)
    pass


def prepare_tasks(project_alias: str):
    """
    tasks need to be according to the TaskCreate model (basically, project=int, data=dict)
    This can be very platform-specific, but the general structure is:
    "text_1", "text_2" (title, description)
    "img_1", ...(image-list of length 2, 1. filtered (grey,blur), 2. original)
    "image_options" (for dynamic choices, where we need the images as options, this need to be included,
    even if empty, when used in the labeling-config, otherwise the UI will break, and not submit)

    Check the data of previous projects, how the urls for the images look like:
    Previously we did something like:    "data/local-files/?d=media/p1_twitter_es/fil/1479181977618112512_0.jpg",
    which needs the images to sit in the right directory of the vm (docker volume)
    but we are moving away from that, because of the addition storage on the vm that is mounted...(see weibo)

    :return:
    """
    from ls_helper.my_labelstudio_client.models import TaskCreate

    po = get_project(alias=project_alias)
    batch_id = int(project_alias.split("_")[1])
    reader = DictReader(Path("data/weibo/basic_posts.csv").open("r", encoding="utf-8"))
    # base_path = Path("data/weibo/2_Weibo data Sample")
    # files = (base_path).glob("*.xlsx")

    # post_hashes = read_data(Path("data/weibo/id_hashes.json"))
    base_img_dir = "https://big5.cssh.bsc.es/MEDIA-WEIBO/batched/"
    tasks = []
    for line in reader:
        if int(line["batch"]) != batch_id:
            continue
        # post_id = line["id"]
        images = json.loads(line["images"])
        task_data = {
            "text_1": line["content"],
            "platform_id": line["hash"],
            "image_options": [f"Image {img + 1}" for img in range(len(images))],
        }
        for idx, img in enumerate(images, start=1):
            task_data[f"img_{idx}"] = [
                base_img_dir + f"filtered/{batch_id}/{line["hash"]}_{img}.jpeg",
                base_img_dir + f"orig/{batch_id}/{line["hash"]}_{img}.jpeg"
            ]
        task = TaskCreate(
            project=po.id,
            data=task_data
        )
        # print(task)
        tasks.append(task)
    po.tasks.import_tasks(LSTaskCreateList(root=tasks))


def main():
    # _create_project()
    prepare_tasks("weibo-ch-5-B_0")


if __name__ == "__main__":
    main()
