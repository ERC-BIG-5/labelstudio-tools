from pytest import fixture

from ls_helper.models.main_models import get_project, ProjectData
from ls_helper.my_labelstudio_client.models import ProjectViewCreate, ProjectViewDataModel, ProjectViewModel

# from pytest.

P_ID = 31


@fixture(autouse=True)
def default_project() -> ProjectData:
    return get_project(P_ID)


def test_create_view(default_project: ProjectData):
    resp = default_project.views.create(ProjectViewCreate(project=P_ID,
                                                          data=ProjectViewDataModel(
                                                              title="test",
                                                          )))
    assert isinstance(resp, ProjectViewModel)
    assert default_project.views.delete(resp.id) == True
