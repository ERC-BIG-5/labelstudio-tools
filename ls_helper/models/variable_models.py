from enum import Enum, auto
from typing import Optional, Literal

from pydantic import BaseModel, model_validator

from tools.project_logging import get_logger

logger = get_logger(__file__)

NO_SINGLE_CHOICE = "NONE"
NO_SINGLE_CHOICE_INDEX = 99


class VariableType(Enum):
    input = auto()
    choice = auto()
    text = auto()
    range = auto()


class VariableModel(BaseModel):
    orig_name: str
    name: str
    type: VariableType
    required: bool = False
    group_variables: Optional[list[str]] = None
    group_name: Optional[str] = None
    group_index: Optional[int] = None


class ChoiceVariableModel(VariableModel):
    orig_options: list[str]
    _all_options: Optional[list[str]] = None
    default: Optional[str]
    choice: Literal["single", "multiple"]  # single, multiple

    # todo, a way to hand in inputs.
    @model_validator(mode="after")
    def all_options(self):
        self._all_options = self.orig_options[:]
        return self

    @property
    def safe_default(self) -> str:
        return self.default or NO_SINGLE_CHOICE

    @property
    def safe_default_index(self) -> str:
        return self.default or NO_SINGLE_CHOICE_INDEX

    @property
    def options(self) -> list[str]:
        return self._all_options

    def option_index(self, val: str) -> int:
        if val in self._all_options:
            return self._all_options.index(val)
        self._all_options.append(val)
        if val != NO_SINGLE_CHOICE:
            logger.warning(f"add value to options list!!! {val}")
        return self._all_options.index(val)
