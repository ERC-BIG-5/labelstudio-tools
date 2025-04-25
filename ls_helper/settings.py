import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings
from tools.env_root import root

if TYPE_CHECKING:
    from ls_helper.my_labelstudio_client.client import LabelStudioBase


class Settings(BaseSettings):
    class Config:
        env_file = ".env"

    LS_HOSTNAME: str
    LS_API_KEY: str
    BASE_DATA_DIR: Optional[Path] = Field(Path("data/ls_data"))

    IN_CONTAINER_LOCAL_STORAGE_BASE: Optional[Path] = Field(None)
    HOST_STORAGE_BASE: Optional[Path] = Field(None)

    DELETED_TASK_FILES_BACKUP_BASE_DIR: Optional[Path] = Field(None)

    __client: Optional["LabelStudioBase"] = None

    @model_validator(mode="after")
    def validate_paths(self, info: ValidationInfo) -> "Settings":
        self.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sub_dirs = {
            "project",
            "variable_extensions",
            "annotations",
            "view",
            "agreements",
            "annotations_results",
            "labeling_configs",
            "plots",
            "tasks",
            "temp",
        }
        for sd in sub_dirs:
            (self.BASE_DATA_DIR / sd).mkdir(parents=True, exist_ok=True)

        return self

    @property
    def client(self) -> "LabelStudioBase":
        from ls_helper.my_labelstudio_client.client import LabelStudioBase

        if self.__client is None:
            self.__client = LabelStudioBase(
                base_url=self.LS_HOSTNAME, api_key=self.LS_API_KEY
            )
        return self.__client

    @property
    def projects_dir(self) -> Path:
        return self.BASE_DATA_DIR / "project"

    @property
    def view_dir(self) -> Path:
        return self.BASE_DATA_DIR / "view"

    @property
    def annotations_dir(self) -> Path:
        return self.BASE_DATA_DIR / "annotations"

    @property
    def annotations_results_dir(self) -> Path:
        return self.BASE_DATA_DIR / "annotations_results"

    @property
    def agreements_dir(self) -> Path:
        return self.BASE_DATA_DIR / "agreements"

    @property
    def labeling_configs_dir(self) -> Path:
        return self.BASE_DATA_DIR / "labeling_configs"

    @property
    def labeling_templates(self) -> Path:
        return self.labeling_configs_dir / "templates"

    @property
    def built_labeling_configs(self) -> Path:
        return self.labeling_configs_dir / "builds"

    @property
    def plots_dir(self) -> Path:
        return self.BASE_DATA_DIR / "plots"

    @property
    def var_extensions_dir(self) -> Path:
        return self.BASE_DATA_DIR / "variable_extensions"

    @property
    def unifix_extensions_file_path(self) -> Path:
        return self.var_extensions_dir / "unifixes.json"

    @property
    def temp_file_path(self) -> Path:
        return self.BASE_DATA_DIR / "temp"

    @property
    def tasks_dir(self) -> Path:
        return self.BASE_DATA_DIR / "tasks"

    @property
    def projects_main_file(self) -> Path:
        return self.BASE_DATA_DIR / "projects.json"


SETTINGS = Settings()
if (root() / ".dev.env").exists():
    DEV_SETTINGS = Settings(_env_file=".dev.env")
else:
    DEV_SETTINGS = None
ls_logger = getLogger("ls-helper")
ls_logger.setLevel(logging.DEBUG)
ls_logger.addHandler(logging.StreamHandler(sys.stdout))
ls_logger.propagate = False


class DFFormat(Enum):
    raw_annotation = auto()
    flat = auto()
    flat_csv_ready = auto()


@dataclass
class DFRawCols:
    CAT: str = "category"
    VAL: str = "value"
    TYPE: str = "type"


@dataclass
class DFCols:
    T_ID: str = "task_id"
    A_ID: str = "annotation_id"
    U_ID: str = "user_id"
    P_ID: str = "platform_id"
    TS: str = "ts"


@dataclass
class AllCols(DFRawCols, DFCols):
    pass


TIMESTAMP_FORMAT = "%Y%m%d_%H%M"
