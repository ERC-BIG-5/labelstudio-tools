import json

from ls_helper.my_labelstudio_client.client import LabelStudioBase
from ls_helper.my_labelstudio_client.models import ProjectModel
from ls_helper.settings import SETTINGS


def create_project(title: str):
    model = ProjectModel.model_validate({"title":title, "created_by":{"id":1}})


    client = LabelStudioBase(base_url=SETTINGS.LS_HOSTNAME, api_key=SETTINGS.LS_API_KEY)
    p = client.get_project(34)
    print(p)
    # json.dump(p.model_dump(),open("test_p.json","w"))
    resp = client.create_project(model)
    print(resp.json())

if __name__ == "__main__":
    create_project("test create")