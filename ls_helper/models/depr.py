from typing import Literal, Optional, Any

from pydantic import BaseModel, Field


class TaskAnnotationItem(BaseModel):
    name: str
    type_: Literal[
        "single",
        "multiple",
        "text",
        "datetime",
        "list-single",
        "list-multiple",
        "list-text",
    ] = Field(None, alias="type")
    num_coders: int
    values: list[list[str]] = Field(default_factory=list)
    value_indices: list[list[int]] = Field(default_factory=list)
    users: list[int | str] = Field(default_factory=list)

    def add(
        self, value: str | list[str], value_index: int | list[int], user_id: int
    ) -> None:
        self.values.append(value)
        self.value_indices.append(value_index)
        self.users.append(user_id)

    def value_str(self, with_defaults: bool = True, with_user: bool = False) -> str:
        # if with_defaults:
        # if with_user:
        #     comb = [f"{v} ({u})" for v, u in zip(self.values, self.users)]
        #     return "; ".join(comb)
        if self.type_.startswith("list"):
            coders = []
            for coder_resp in self.values:
                # for item in coder_resp:
                #     items.append(["|".join(item) for item in item])
                coders.append(["|".join(item) for item in coder_resp])
            coder_join = [", ".join(cv) for cv in coders]
        else:
            coder_join = [", ".join(cv) for cv in self.values]
        return "; ".join(coder_join)


class TaskAnnotResults(BaseModel):
    task_id: int
    items: Optional[dict[str, TaskAnnotationItem]] = Field(default_factory=dict)
    num_coders: int
    num_cancellations: int
    relevant_input_data: dict[str, Any]
    users: list[int]

    def add(
        self,
        item_name: str,
        value: list[str] | list[list[str]],
        value_index: list[int] | list[list[int]],
        user_id: int,
        type_: Literal[
            "single",
            "multiple",
            "text",
            "datetime",
            "list-single",
            "list-multiple",
            "list-text",
        ],
    ) -> None:
        annotation_item = self.items.setdefault(
            item_name,
            TaskAnnotationItem.model_validate(
                {"name": item_name, "type": type_, "num_coders": self.num_coders}
            ),
        )

        annotation_item.add(value, value_index, user_id)

    def data(self) -> dict[str, Any]:
        return {k: v.values for k, v in self.items.items()}

    def data_str(self, with_defaults: bool = True) -> dict[str, str]:
        return {k: v.value_str(with_defaults) for k, v in self.items.items()}

    class Config:
        validate_assignment = True

    def set_all_to_pre_default(self) -> None:
        for res in self.items.values():
            res.set_predefaults()
