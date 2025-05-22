"""
Use os.chdir so we can just run main here.

_create_tasks is interesting cuz it shows off how to copy tasks from one p to another
(while filtering only relevant tasks, coded by one as relevant...

"""
from ls_helper.models.main_models import get_project
from ls_helper.my_labelstudio_client.models import TaskCreateList, TaskCreate


def _create_project():
    from ls_helper.command.project_setup import create_project
    create_project("Youtube EN - p.v6", "youtube-en-6", "youtube", "en", 2)
    pass


def main():
    # _create_project()
    _create_tasks()


def _create_tasks():
    po4 = get_project(**{"alias": "youtube-en-4"})

    # get the raw df
    raw_ann_df = po4.get_annotations_results().raw_annotation_df
    # get all nature_any responses
    nature_any_resp = raw_ann_df[raw_ann_df["variable"] == "nature_any"]
    # turn those value lists (value is always a list) into strings
    nature_any_resp = nature_any_resp.explode("value")
    # get the positive response
    nature_any_resp_yes = nature_any_resp[nature_any_resp["value"] == "Yes"]
    nature_platform_ids = set(nature_any_resp_yes["platform_id"].to_list())

    # now get all tasks and filter them:
    tasks = po4.tasks.get()
    relevant_tasks = [
        t for t in tasks.root
        if t.data["platform_id"] in nature_platform_ids
    ]
    # lets check
    assert len(relevant_tasks) == len(nature_platform_ids)
    # transform them. this makes sure that, the id is not passed to the client_function, cuz that will break it there
    create_tasks = TaskCreateList(root=[TaskCreate.model_validate(t.model_dump()) for t in relevant_tasks])

    po6 = get_project(**{"alias": "youtube-en-6"})
    po6.tasks.import_tasks(create_tasks)


if __name__ == "__main__":
    # agreements(alias="youtube-en-4")
    main()
