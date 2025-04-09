import enum
import os
import typing
from functools import lru_cache
from typing import Optional
import json
from pathlib import Path
from datetime import datetime
import httpx
from httpx import Response

from ls_helper.my_labelstudio_client.models import ProjectViewModel, ProjectModel, UserModel, ProjectViewCreate, \
    TaskCreate, TaskResultModel
from ls_helper.settings import SETTINGS, DEV_SETTINGS

if typing.TYPE_CHECKING:
    from ls_helper.models import ProjectAnnotations


class LabelStudioEnvironment(enum.Enum):
    DEFAULT = "http://localhost:8080"


class ApiError(Exception):
    status_code: Optional[int]
    body: typing.Any

    def __init__(self, *, status_code: Optional[int] = None, body: typing.Any = None):
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"status_code: {self.status_code}, body: {self.body}"


class BaseClientWrapper:
    def __init__(self, *, api_key: str, base_url: str, timeout: Optional[float] = None):
        self.api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    def get_headers(self) -> typing.Dict[str, str]:
        headers: typing.Dict[str, str] = {
            "X-Fern-Language": "Python",
            "X-Fern-SDK-Name": "label-studio-sdk",
            "X-Fern-SDK-Version": "1.0.8",
        }
        headers["Authorization"] = f"Token  {self.api_key}"
        return headers

    def get_base_url(self) -> str:
        return self._base_url

    def get_timeout(self) -> Optional[float]:
        return self._timeout


class SyncClientWrapper(BaseClientWrapper):
    def __init__(
            self, *, api_key: str, base_url: str, timeout: Optional[float] = None, httpx_client: httpx.Client
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self.httpx_client = httpx.Client(
            headers=self.get_headers(),
            timeout=self.get_timeout(),
            base_url=self.get_base_url(),
        )


class LabelStudioBase:
    """
    Use this class to access the different functions within the SDK. You can instantiate any number of clients with different configuration that will propagate to these functions.

    Parameters
    ----------
    base_url : Optional[str]
        The base url to use for requests from the client.

    environment : LabelStudioEnvironment
        The environment to use for requests from the client. from .environment import LabelStudioEnvironment



        Defaults to LabelStudioEnvironment.DEFAULT



    api_key : Optional[str]
    timeout : Optional[float]
        The timeout to be used, in seconds, for requests. By default the timeout is 60 seconds, unless a custom httpx client is used, in which case this default is not enforced.

    follow_redirects : Optional[bool]
        Whether the default httpx client follows redirects or not, this is irrelevant if a custom httpx client is passed in.

    httpx_client : Optional[httpx.Client]
        The httpx client to use for making requests, a preconfigured client is used by default, however this is useful should you want to pass in any custom httpx configuration.

    Examples
    --------
    from label_studio_sdk.client import LabelStudio

    client = LabelStudio(
        api_key="YOUR_API_KEY",
    )
    """

    def __init__(
            self,
            *,
            base_url: Optional[str] = None,
            environment: LabelStudioEnvironment = LabelStudioEnvironment.DEFAULT,
            api_key: Optional[str] = os.getenv("LABEL_STUDIO_API_KEY"),
            timeout: Optional[float] = None,
            follow_redirects: Optional[bool] = True,
            httpx_client: Optional[httpx.Client] = None
    ):
        _defaulted_timeout = timeout if timeout is not None else 60 if httpx_client is None else None
        if api_key is None:
            raise ApiError(
                body="The client must be instantiated be either passing in api_key or setting LABEL_STUDIO_API_KEY"
            )
        self._client_wrapper = SyncClientWrapper(
            base_url=base_url,
            api_key=api_key,
            httpx_client=httpx_client
            if httpx_client is not None
            else httpx.Client(timeout=_defaulted_timeout, follow_redirects=follow_redirects)
            if follow_redirects is not None
            else httpx.Client(timeout=_defaulted_timeout),
            timeout=_defaulted_timeout,
        )
        """
        self.annotations = AnnotationsClient(client_wrapper=self._client_wrapper)
        self.users = UsersClient(client_wrapper=self._client_wrapper)
        self.actions = ActionsClient(client_wrapper=self._client_wrapper)
        self.views = ViewsClient(client_wrapper=self._client_wrapper)
        self.files = FilesClient(client_wrapper=self._client_wrapper)
        self.projects = ProjectsClient(client_wrapper=self._client_wrapper)
        self.ml = MlClient(client_wrapper=self._client_wrapper)
        self.predictions = PredictionsClient(client_wrapper=self._client_wrapper)
        self.tasks = TasksClient(client_wrapper=self._client_wrapper)
        self.import_storage = ImportStorageClient(client_wrapper=self._client_wrapper)
        self.export_storage = ExportStorageClient(client_wrapper=self._client_wrapper)
        self.webhooks = WebhooksClient(client_wrapper=self._client_wrapper)
        self.prompts = PromptsClient(client_wrapper=self._client_wrapper)
        self.model_providers = ModelProvidersClient(client_wrapper=self._client_wrapper)
        self.comments = CommentsClient(client_wrapper=self._client_wrapper)
        self.workspaces = WorkspacesClient(client_wrapper=self._client_wrapper)

        """

    def projects_list(self) -> list[ProjectModel]:
        resp = self._client_wrapper.httpx_client.get("/api/projects")
        if resp.status_code == 200:
            return [ProjectModel.model_validate(p) for p in resp.json()["results"]]

    def get_project(self, project_id) -> ProjectModel:
        resp = self._client_wrapper.httpx_client.get("/api/projects/{}".format(project_id))
        if resp.status_code == 200:
            return ProjectModel.model_validate(resp.json())

    def patch_project(self, project_id: int, data: dict) -> Optional[ProjectModel]:
        resp = self._client_wrapper.httpx_client.patch(f"/api/projects/{project_id}", json=data)
        if resp.status_code == 200:
            return ProjectModel.model_validate(resp.json())
        else:
            print(resp.status_code, resp.json())

    def get_project_annotations(self, project_id: int) -> list[TaskResultModel]:
        export_create = self._client_wrapper.httpx_client.post(f"/api/projects/{project_id}/exports", json={
            "task_filter_options": {"only_with_annotations": True}
        })
        export_data = export_create.json()
        export_id = export_data["id"]

        dl = self._client_wrapper.httpx_client.get(f"api/projects/{project_id}/exports/{export_id}/download")

        result = dl.json()

        return [TaskResultModel.model_validate(t) for t in result]

    def get_project_views(self, project_id: int) -> list[ProjectViewModel]:
        resp = self._client_wrapper.httpx_client.get(f"api/dm/views/?project={project_id}")
        return [ProjectViewModel.model_validate(v) for v in resp.json()]

    def get_users(self, dump: bool = True):
        resp = self._client_wrapper.httpx_client.get("/api/users")
        users = list(map(UserModel.model_validate, resp.json()))
        if dump:
            json.dump(list(map(UserModel.model_dump, users)), open("data/users.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        return users

    def get_user(self, user_id: int) -> Optional[UserModel]:
        resp = self._client_wrapper.httpx_client.get(f"api/users/{user_id}")
        if resp.status_code == 200:
            return UserModel.model_validate(resp.json())

    def update_user(self, user: UserModel):
        resp = self._client_wrapper.httpx_client.patch(f"/api/users/{user.id}", json=user.model_dump(exclude={"email"}))
        print(resp.status_code, resp.text)

    def create_project(self, data: ProjectModel):
        resp = self._client_wrapper.httpx_client.post(f"/api/projects/", json=data.model_dump(exclude_defaults=True))
        if resp.status_code != 201:
            raise ValueError(f"failed to create project: {resp.status_code}\n{resp.json()}")
        return ProjectModel.model_validate(resp.json())

    def patch_view(self, view_id: int, data: dict):
        resp = self._client_wrapper.httpx_client.patch(f"/api/dm/views/{view_id}", json=data)
        return resp

    def get_task(self, task_id: int):
        resp = self._client_wrapper.httpx_client.get(f"/api/tasks/{task_id}")
        return resp

    def get_task_list(self,
                      *,
                      page: Optional[int] = None,
                      page_size: Optional[int] = 2000,
                      project: Optional[int] = None,
                      view: Optional[int] = None,
                      resolve_url: Optional[bool] = False,
                      fields: Optional[typing.Literal["all", "task_only"]] = "all",
                      review: Optional[bool] = None,
                      include: Optional[str] = None,
                      query: Optional[str] = None) -> Response:
        # https://api.labelstud.io/api-reference/api-reference/tasks/list
        params = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        if project is not None:
            params["project"] = project
        if view is not None:
            params["view"] = view
        if resolve_url is not None:
            params["resolve_url"] = resolve_url
        if fields is not None:
            params["fields"] = fields
        if review is not None:
            params["review"] = review
        if include is not None:
            params["include"] = include
        if query is not None:
            params["query"] = query
        return self._client_wrapper.httpx_client.get("/api/tasks", params=params, timeout=60)

    def delete_task(self, task_id: int):
        pass

    def patch_task(self, task_id: int, data: dict):
        resp = self._client_wrapper.httpx_client.patch(f"/api/tasks/{task_id}", json=data)
        return resp

    def list_import_storages(self, project: Optional[int] = None) -> Response:
        resp = self._client_wrapper.httpx_client.get("api/storages/localfiles/", params={"project": project})
        return resp

    def validate_project_labeling_config(self, p_id: int, label_config: str):
        resp = self._client_wrapper.httpx_client.post(f"/api/projects/{p_id}/validate/",
                                                      json={"label_config": label_config})
        return resp

    def create_view(self, data: ProjectViewCreate) -> ProjectViewModel:
        resp = self._client_wrapper.httpx_client.post(f"/api/dm/views/", json=data.model_dump(exclude_defaults=True))
        if resp.status_code != 201:
            print(resp)
            raise ValueError(resp.json())
        view = ProjectViewModel.model_validate(resp.json())
        return view

    def create_task(self, data: TaskCreate):
        resp = self._client_wrapper.httpx_client.post(f"/api/tasks/", json=data.model_dump())
        if resp.status_code != 201:
            print(resp)
            raise ValueError(resp.json())
        return resp

    def import_tasks(self, project_id: int, tasks: list[TaskCreate]):
        resp = self._client_wrapper.httpx_client.post(f"/api/projects/{project_id}/import",
                                                      json=[t.model_dump()["data"] for t in tasks])
        if resp.status_code != 201:
            print(resp)
            raise ValueError(resp.json())
        return resp

    def delete_view(self, view_id: int):
        resp = self._client_wrapper.httpx_client.delete(f"/api/dm/views/{view_id}")
        print(resp)

_GLOBAL_CLIENT: Optional[LabelStudioBase] = None


@lru_cache
def ls_client(dev: Optional[bool] = None, ignore_global_client: bool = False) -> LabelStudioBase:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT and not ignore_global_client:
        return _GLOBAL_CLIENT

    if dev is None:
        dev = False
    settings = DEV_SETTINGS if dev else SETTINGS

    client = LabelStudioBase(base_url=settings.LS_HOSTNAME, api_key=settings.LS_API_KEY)
    _GLOBAL_CLIENT = client
    print(f"global client set to [{dev=}]")
    return client
