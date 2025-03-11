from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings

from ls_helper.my_labelstudio_client.client import LabelStudioBase


class Settings(BaseSettings):
    class Config:
        env_file = ".env"

    LS_HOSTNAME: str
    LS_API_KEY: str
    BASE_DATA_DIR: Optional[Path] = Field(Path("data/ls_data"))

    IN_CONTAINER_LOCAL_STORAGE_BASE: Optional[Path] = Field(None)
    HOST_STORAGE_BASE: Optional[Path] = Field(None)

    __client : Optional[LabelStudioBase] = None


    @model_validator(mode="after")
    def validate_paths(self, info: ValidationInfo) -> dict:
        self.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sub_dirs = {"project", "fixes", "annotations", "view"}
        for sd in sub_dirs:
            (self.BASE_DATA_DIR / sd).mkdir(parents=True, exist_ok=True)

        return self


    @property
    def client(self) -> LabelStudioBase:
        if self.__client is None:
            self.__client = LabelStudioBase(base_url=self.LS_HOSTNAME, api_key=self.LS_API_KEY)
        return self.__client

    @property
    def projects_dir(self) -> Path:
        return self.BASE_DATA_DIR / "project"

SETTINGS = Settings()

if __name__ == "__main__":
    print(SETTINGS.projects.youtube.en.id)
