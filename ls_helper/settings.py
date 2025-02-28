from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    class Config:
        env_file = ".env"

    LS_HOSTNAME: str
    LS_API_KEY: str
    BASE_DATA_DIR: Optional[Path] = Field(default_factory=lambda _: Path("./data"))


SETTINGS = Settings()
