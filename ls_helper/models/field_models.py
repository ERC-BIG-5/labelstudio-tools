from enum import auto, Enum
from typing import Optional

from pydantic import BaseModel

class FieldType(Enum):
    input= auto()
    choice= auto()
    text= auto()

class FieldModel(BaseModel):
    orig_name: str
    name: str
    type: FieldType
    required: bool = False

class ChoiceFieldModel(FieldModel):
    options: list[str]
    default: Optional[str]
    choice: str # single, multiple

    # todo, a way to hand in inputs.