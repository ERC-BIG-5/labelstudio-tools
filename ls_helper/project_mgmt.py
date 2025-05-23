import json
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Any, Optional

import orjson
from tools.env_root import root
from tools.files import read_data

from ls_helper.funcs import get_latest_annotation_file
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.my_labelstudio_client.models import (
    ProjectModel,
    ProjectViewCreate,
    ProjectViewModel,
    TaskResultModel,
)
from ls_helper.models.main_models import (
    ProjectAnnotationResultsModel,
    ProjectCreate,
    ProjectData,
    platforms_overview,
)
from ls_helper.settings import SETTINGS, TIMESTAMP_FORMAT, ls_logger


class FileType(Enum):
    LSProject = auto()
    LSAnnotation = auto()
    LSView = auto()
    Extension = auto()


class ProjectMgmt:
    DEFAULT_VIEW_HIDDEN_COLUMNS_FP = (
        root() / "data/ls_data/default/ls_project_view_hiddenColumns.json"
    )

    @staticmethod
    def default_project_values() -> dict[str, Any]:
        return {
            "color": "#617ada",
            "maximum_annotations": 2,
            "sampling": "Uniform sampling",
            "show_collab_predictions": True,
        }

    @staticmethod
    def get_latest_file(p_id: int, file_type: FileType) -> Optional[Path]:
        raise NotImplementedError

    @staticmethod
    def update_projects():
        """
        TODO, not sure if still used or required.
        :return:
        """
        projects_data = SETTINGS.client.projects_list()
        project_map = {p.id: p for p in projects_data}

        projects_info = platforms_overview
        for platform, p_data in projects_info:
            # print(platform, p_data)
            for lang, l_d in p_data.items():
                # print(lang, l_d)
                if _id := l_d.id:
                    if _id not in project_map:
                        print(
                            f"warning: project-info has {platform}.{lang} with id: {_id} but does not exist in LS data"
                        )
                    dest = p_data.project_data_path(platform, lang)
                    ls_project_data = project_map[_id]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(
                        orjson.dumps(
                            ls_project_data.model_dump(),
                            option=orjson.OPT_INDENT_2,
                        ).decode("utf-8"),
                        encoding="utf-8",
                    )

    @classmethod
    def create_view(cls, view: ProjectViewCreate) -> ProjectViewModel:
        if not view.data.hiddenColumns:
            view.data.hiddenColumns = read_data(
                cls.DEFAULT_VIEW_HIDDEN_COLUMNS_FP
            )
        return ls_client().create_view(view)

    @classmethod
    def refresh_views(self, po: ProjectData):
        views = ls_client().get_project_views(po.id)
        po.view_file.write_text(json.dumps([v.model_dump() for v in views]))

    @classmethod
    def create_project(
        cls, p: ProjectCreate, add_coding_game_view: bool = True
    ) -> tuple[ProjectModel, Optional[ProjectViewModel]]:
        model = ProjectModel(
            title=p.title,
            description=p.full_description,
            **ProjectMgmt.default_project_values(),
        )
        project = ls_client().create_project(model)
        coding_game_view: Optional[ProjectViewModel] = None
        if add_coding_game_view:
            coding_game_view = cls.create_view(
                ProjectViewCreate.model_validate(
                    {"project": project.id, "data": {}}
                )
            )

        return project, coding_game_view

    @staticmethod
    def get_recent_annotations(
        project_id: int,
        accepted_age: int,
        use_existing: bool = False,
    ) -> Optional[
        tuple[
            Annotated[bool, "use_local"],
            Optional[ProjectAnnotationResultsModel],
        ]
    ]:
        """

        :param project_id:
        :param accepted_age:
        :return: true, list of anns; True if existing file
        """
        # todo change the list back to another model in order to pack some functions like dropping cancelations...
        latest_file = get_latest_annotation_file(project_id)
        if latest_file is not None:
            file_dt = datetime.strptime(latest_file.stem, "%Y%m%d_%H%M")
            # print(file_dt, datetime.now(), datetime.now() - file_dt)
            if (
                datetime.now() - file_dt < timedelta(hours=accepted_age)
                or use_existing
            ):
                ls_logger.info(
                    f"Get recent, gets latest annotation: {file_dt:%m%d_%H%M}"
                )

                annotation_file = get_latest_annotation_file(project_id)
                if not annotation_file:
                    ls_logger.warning("No annotation file?!")
                    return None
                task_results = [
                    TaskResultModel.model_validate(t)
                    for t in json.load(annotation_file.open(encoding="utf-8"))
                ]

                return True, ProjectAnnotationResultsModel(
                    task_results=task_results, timestamp=file_dt
                )

        # todo this stuff is old. needs refactoring love. move to ProjectData model
        print("downloading annotations")
        result = ls_client().get_project_annotations(project_id)
        if not result:
            return False, None
        ts = datetime.now()
        res_path = (
            SETTINGS.annotations_dir
            / str(project_id)
            / f"{ts.strftime(TIMESTAMP_FORMAT)}.json"
        )
        res_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"dumping project annotations to {res_path}")
        pa = ProjectAnnotationResultsModel(task_results=result, timestamp=ts)
        json.dump(
            [r.model_dump() for r in result],
            res_path.open("w", encoding="utf-8"),
        )
        return False, pa
