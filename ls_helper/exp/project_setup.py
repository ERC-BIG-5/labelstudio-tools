"""
guideline for setting up a project (example Weibo)
"""
from pathlib import Path


def _create_project():
    from ls_helper.command.project_setup import create_project
    create_project("Weibo p.v5", "weibo-ch-5", "weibo", "ch", 2)
    pass


def prepare_tasks():
    """
    tasks need to be according to the TaskCreate model (basically, project=int, data=dict)
    :return:
    """
    from ls_helper.my_labelstudio_client.models import TaskCreate
    TaskCreate

    base_path = Path("data/weibo/2_Weibo data Sample")
    files = (base_path).glob("*.xlsx")


def main():
    # _create_project()
    prepare_tasks()


if __name__ == "__main__":
    main()
