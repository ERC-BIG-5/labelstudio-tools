from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, Any, Iterable, Annotated

from deprecated.classic import deprecated
from pydantic import BaseModel, Field, model_validator, PlainSerializer

from ls_helper.models.field_models import FieldType
from tools import fast_levenhstein
from tools.project_logging import get_logger

# todo bring and import tools,
SerializableDatetime = Annotated[
    datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str, when_used='json')
]

SerializableDatetimeAlways = Annotated[
    datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str, when_used='always')
]

PlLang = tuple[str, str]
# ProjectAccess = int | str | PlLang

logger = get_logger(__file__)


class IField(BaseModel):
    pass


class AField(IField):
    name: str


class IChoice(BaseModel):
    value: str
    alias: Optional[str] = None

    @property
    def annot_val(self) -> str:
        if self.alias:
            return self.alias
        return self.value


class ChoicesType(str, Enum):
    single = "single"
    multiple = "multiple"


class IChoices(AField):
    toName: str
    options: list[IChoice]
    choice: ChoicesType = "single"
    indices: Optional[list[str]] = Field(default_factory=list)
    value: Optional[str] = None # thats when the options come from the data

    @model_validator(mode="after")
    def create_indices(cls, data: "IChoices"):
        data.indices = [c.annot_val for c in data.options]
        return data

    def get_index(self, value: str | list[str]) -> int | list[int]:
        if isinstance(value, str):
            return self.indices.index(value)
        else:
            return [self.indices.index(v) for v in value]

    def insert_option(self, index, choice: IChoice):
        self.options.insert(index, choice)
        self.indices = [c.annot_val for c in self.options]

    def raw_options_list(self) -> list[str]:
        return [c.annot_val for c in self.options]


class ITextArea(AField):
    toName: Optional[str] = None
    required: Optional[bool] = False


class IText(IField):
    value: Optional[str] = None


class InterfaceData(BaseModel):
    ordered_fields_map: dict[str, IField]= Field(default_factory=dict)

    #
    orig_choices: dict[str, IChoices] = Field(default_factory=dict)
    # choices: Optional[dict[str, IChoices]] = Field(default_factory=dict)
    # this should have a model, required, indexed, ...
    # free_text: list[str]
    #
    inputs: dict[str, str] = Field(description="Map from el.name > el.value")
    _extension_applied: bool = False

    @property
    def free_text(self) -> list[str]:
        oa = self.ordered_fields_map.items()
        return list(map(lambda f: f[0],
                        filter(lambda f: isinstance(f[1], IText), oa)))

    def find_name_fixes(self, orig_keys: Iterable[str],
                        name_fixes: dict[str, str],
                        report_missing: bool = False) -> dict[str, str]:
        result: dict[str, str] = {}
        for k in orig_keys:
            if new_name := name_fixes.get(k):
                result[k] = new_name
            elif report_missing:
                print(f"Missing name.fix for {k}")

        return result

    def apply_extension(self,
                        data_extensions: "ProjectFieldsExtensions",
                        allow_non_existing_defaults: bool = True):
        if self._extension_applied:
            return
        name_fixes = data_extensions.name_fixes
        ordered_name_fixes = self.find_name_fixes(self.ordered_fields, name_fixes)
        for old, new in ordered_name_fixes.items():
            self.ordered_fields[self.ordered_fields.index(old)] = new
        choices_name_fixes = self.find_name_fixes(self.orig_choices.keys(), name_fixes, True)

        for old, new in choices_name_fixes.items():
            self.choices[new] = self.orig_choices[old]

        # check if defaults are correct
        for k, v in self.orig_choices.items():
            ext = data_extensions.extensions[v.name]
            # catch non-existing defaults...
            if ext:
                if ext.default:
                    if not allow_non_existing_defaults:
                        if v.choice == "single":
                            if not isinstance(ext.default, str):
                                raise ValueError(f"Choice {k} has default value {ext.default}")
                            if ext.default not in v.raw_options_list():
                                raise ValueError(
                                    f"Choice {k} has default invalid value {ext.default}, options: {v.raw_options_list()}")
                        elif v.choice == "multiple":
                            if not isinstance(ext.default, list):
                                raise ValueError(f"Choice {k} has default value {ext.default}")
                            if any(d not in v.raw_options_list() for d in ext.default):
                                raise ValueError(
                                    f"Choice {k} has default invalid value {ext.default}, options: {v.raw_options_list()}")
                    # TODO pass actually add the default...
                    v.insert_option(0, IChoice(value=ext.default, alias=ext.default))
            else:
                field_extensions = list(data_extensions.extension_reverse_map.keys())
                logger.error(
                    f"Choice '{k}' has no extension:. Maybe...: >>> {(fast_levenhstein.levenhstein_get_closest_matches(k, field_extensions))}"
                    f"Download the latest project-data(or check it on the page). check it against your project-data."
                    f"Fix your extensions.")

        text_name_fixes = self.find_name_fixes(self.free_text, name_fixes, True)
        for old, new in text_name_fixes.items():
            self.free_text[self.free_text.index(old)] = new
        self._extension_applied = True

    def field_type(self, q) -> FieldType:
        """
        in some parts, we turn column indices into $, so this is the
        way to get the original type
        # todo, not good approach. we should have the column merging
        # as a flag and store the original type with it
        :param q:
        :return:
        """
        field = self.ordered_fields_map.get(q)
        if not field:
            # not required when catching during validation...
            raise ValueError(f"Field {q} has not been defined")
        if isinstance(field,IChoices):
            return FieldType.choice
        elif isinstance(field, IText):
            return FieldType.text
        elif isinstance(field, ITextArea):
            return FieldType.text
        else:
        # if "$" in q:
        #     q = q.replace("$", "0")
        # if q in self.orig_choices:
        #     return FieldType.choice
        # elif q in self.free_text:
        #     return FieldType.text
        # else:
            logger.error(f"unknown question type for {q}: {type(field)}. defaulting to text")
            return FieldType.text

    def __contains__(self, item):
        return item in self.ordered_fields

    @property
    def choices(self) -> dict[str, IChoices]:
        oa = self.ordered_fields_map.items()
        return dict(
            filter(lambda f: isinstance(f[1], IChoices), oa))

    @property
    def ordered_fields(self) -> list[str]:
        return list(self.ordered_fields_map.keys())


class FieldExtension(BaseModel):
    name_fix: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str | list[str]] = None
    deprecated: Optional[bool] = None


class ProjectFieldsExtensions(BaseModel):
    extensions: dict[str, FieldExtension] = Field(alias="fixes")
    extension_reverse_map: dict[str, str] = Field(description="fixes[k].name_fix = fixes[k]", default_factory=dict,
                                                  exclude=True)

    def model_post_init(self, __context: Any) -> None:
        for k, v in self.extensions.items():
            if v.name_fix:
                self.extension_reverse_map[v.name_fix] = k
            else:
                self.extension_reverse_map[k] = k

    def get_from_new_name(self, new_name: str) -> Optional[FieldExtension]:
        orig_name = self.extension_reverse_map.get(new_name)
        if orig_name:
            return self.extensions[orig_name]

    @property
    def name_fixes(self) -> dict[str, str]:
        return {f: d.name_fix or f for f, d in self.extensions.items()}


class PrincipleRow(BaseModel):
    task_id: int
    ann_id: int
    user_id: int
    # user: Optional[str] = None
    platform_id: str
    ts: datetime
    type: str
    category: str
    value: list[str]


class FullAnnotationRow(BaseModel):
    task_id: int
    ann_id: int
    platform_id: str
    user_id: int
    user: Optional[str] = None
    updated_at: datetime
    results: dict[str, Optional[list[str]]] = Field(default_factory=dict)
