import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, RootModel

from tools.pydantic_annotated_types import SerializableDatetimeAlways


class ProjectModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: Optional[int] = None
    title: str
    description: Optional[str] = ""
    label_config: Optional[str] = "<View></View>"
    expert_instruction: Optional[str] = ""
    show_instruction: Optional[bool] = False

    show_skip_button: Optional[bool] = Field(default=False)
    """
    Show a skip button in interface and allow annotators to skip the task
    """

    enable_empty_annotation: Optional[bool] = Field(default=True)
    """
    Allow annotators to submit empty annotations
    """

    show_annotation_history: Optional[bool] = Field(default=True)
    """
    Show annotation history to annotator
    """

    organization: Optional[int] = None
    color: Optional[str] = "#FFFFFF"
    maximum_annotations: Optional[int] = Field(default=4)
    """
    Maximum number of annotations for one task. If the number of annotations per task is equal or greater to this value, the task is completed (is_labeled=True)
    """

    is_published: Optional[bool] = Field(default=True)
    """
    Whether or not the project is published to annotators
    """

    model_version: Optional[str] = Field(default=None, alias="model_version")
    """
    Machine learning model version
    """

    is_draft: Optional[bool] = Field(default=False)
    """
    Whether or not the project is in the middle of being created
    """

    # TODO
    created_by: Optional[dict] = None  # UserSimple
    created_at: Optional[datetime] = None
    min_annotations_to_start_training: Optional[int] = Field(default=1)
    """
    Minimum number of completed tasks after which model training is started
    """

    start_training_on_annotation_update: Optional[str | bool] = Field(
        default=False
    )
    """
    Start model training after any annotations are submitted or updated
    """

    show_collab_predictions: Optional[bool] = Field(default=False)
    """
    If set, the annotator can view model predictions
    """

    num_tasks_with_annotations: Optional[int] = Field(default=0)
    """
    Tasks with annotations count
    """

    task_number: Optional[int] = Field(default=None)
    """
    Total task number in project
    """

    useful_annotation_number: Optional[int] = Field(default=None)
    """
    Useful annotation number in project not including skipped_annotations_number and ground_truth_number. Total annotations = annotation_number + skipped_annotations_number + ground_truth_number
    """

    ground_truth_number: Optional[int] = Field(default=None)
    """
    Honeypot annotation number in project
    """

    skipped_annotations_number: Optional[int] = Field(default=None)
    """
    Skipped by collaborators annotation number in project
    """

    total_annotations_number: Optional[int] = Field(default=None)
    """
    Total annotations number in project including skipped_annotations_number and ground_truth_number.
    """

    total_predictions_number: Optional[int] = Field(default=None)
    """
    Total predictions number in project including skipped_annotations_number, ground_truth_number, and useful_annotation_number.
    """

    # todo
    sampling: Optional[dict | str] = "Uniform sampling"  # ProjectSampling
    show_ground_truth_first: Optional[bool] = False
    show_overlap_first: Optional[bool] = False
    overlap_cohort_percentage: Optional[int] = 100
    task_data_login: Optional[str] = Field(default=None)
    """
    Task data credentials: login
    """

    task_data_password: Optional[str] = Field(default=None)
    """
    Task data credentials: password
    """

    control_weights: Optional[dict[str, Any]] = Field(default=None)
    """
    Dict of weights for each control tag in metric calculation. Each control tag (e.g. label or choice) will have it's own key in control weight dict with weight for each label and overall weight.For example, if bounding box annotation with control tag named my_bbox should be included with 0.33 weight in agreement calculation, and the first label Car should be twice more important than Airplaine, then you have to need the specify: {'my_bbox': {'type': 'RectangleLabels', 'labels': {'Car': 1.0, 'Airplaine': 0.5}, 'overall': 0.33}
    """

    parsed_label_config: Optional[dict[str, Any]] = Field(default=None)
    """
    JSON-formatted labeling configuration
    """

    evaluate_predictions_automatically: Optional[bool] = Field(default=False)
    """
    Retrieve and display predictions when loading a task
    """

    config_has_control_tags: Optional[str | bool] = Field(default=None)
    """
    Flag to detect is project ready for labeling
    """
    # todo
    skip_queue: Optional[dict | str] = "REQUEUE_FOR_OTHERS"  # ProjectSkipQueue
    reveal_preannotations_interactively: Optional[bool] = Field(default=False)
    """
    Reveal pre-annotations interactively
    """

    pinned_at: Optional[datetime] = Field(default=None)
    """
    Pinned date and time
    """

    finished_task_number: Optional[int] = Field(default=None)
    pass


class HiddenColumns(BaseModel):
    explore: Optional[list["str"]] = Field(default_factory=list)
    labeling: Optional[list["str"]] = Field(default_factory=list)


class ViewFilterItem(BaseModel):
    filter: str
    operator: str
    type: str
    value: str | int | float | list[str] | list[int] | list[float]


class ViewFilter(BaseModel):
    conjunction: Literal["and", "or"] = Field(default="and")
    items: list[ViewFilterItem] = Field(default_factory=list)


# from LS
class ProjectViewDataModel(BaseModel):
    title: str = None
    hiddenColumns: Optional[HiddenColumns] = Field(
        default_factory=HiddenColumns
    )
    type: Optional[str] = None
    target: Optional[str] = None
    gridWidth: Optional[int] = None
    columnWidth: Optional[int] = None

    semantic_search: Optional[list] = None
    columnsDisplay: Optional[dict] = None
    filters: Optional[ViewFilter] = Field(default_factory=ViewFilter)
    ordering: Optional[list[str]] = None


class ProjectViewCreate(BaseModel):
    project: int
    data: Optional[ProjectViewDataModel] = Field(
        default_factory=ProjectViewDataModel
    )


class ViewType(str, Enum):
    conflict = "conflict"
    coding_game = "coding-game"


class TaskCreate(BaseModel):
    project: int
    data: Optional[dict[str, Any]] = None
    predictions: Optional[list] = None

    # model_config = ConfigDict(extra="allow")


class Task(TaskCreate):
    id: int
    # todo, there is actually much more...

    model_config = ConfigDict(extra="allow")


class TaskCreateList(RootModel):
    root: list[TaskCreate]


class TaskList(RootModel):
    root: list[Task]


# from LS
class ProjectViewModel(ProjectViewCreate):
    id: int
    user: int
    order: int


class UserModel(BaseModel):
    id: int
    first_name: str
    last_name: str
    username: str
    email: str
    last_activity: Annotated[
        datetime,
        PlainSerializer(
            lambda dt: dt.isoformat(), return_type=str, when_used="always"
        ),
    ]
    avatar: Optional[str] = None
    initials: str
    phone: Optional[str] = None
    active_organization: int
    allow_newsletters: Optional[bool] = False
    # date_joined: datetime


class ViewModel(BaseModel):
    id: int
    order: int
    user: int
    project: int
    data: ProjectViewDataModel


class ChoicesValue(BaseModel):
    choices: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.choices))

    @property
    def direct_value(self) -> list[str]:
        return self.choices


class TextValue(BaseModel):
    text: list[str]

    @property
    def str_value(self) -> str:
        return str(",".join(self.text))

    @property
    def direct_value(self) -> list[str]:
        return self.text


class TimelineLabelsRange(BaseModel):
    start: int
    end: int

    @property
    def str_value(self) -> str:
        return f"{self.start}-{self.end}"


class TimelineLabels(BaseModel):
    ranges: list[TimelineLabelsRange]
    timelinelabels: list[str]

    @property
    def direct_value(self) -> list[str]:
        return [
            f'{",".join(self.timelinelabels)}:{"".join([r.str_value for r in self.ranges])}'
        ]


class AnnotationResult(BaseModel):
    id: str
    type: str
    value: ChoicesValue | TextValue | TimelineLabels
    origin: str
    to_name: str
    from_name: str

    @property
    def str_value(self) -> str:
        return self.value.str_value

    @property
    def direct_value(self) -> list[str]:
        return self.value.direct_value


class TaskAnnotationModel(BaseModel):
    id: int
    completed_by: int
    result: list[AnnotationResult]
    was_cancelled: bool
    ground_truth: bool
    created_at: SerializableDatetimeAlways
    updated_at: SerializableDatetimeAlways
    draft_created_at: Optional[SerializableDatetimeAlways] = None
    lead_time: float
    prediction: dict
    result_count: int
    unique_id: Annotated[
        uuid.UUID,
        PlainSerializer(lambda v: str(v), return_type=str, when_used="always"),
    ]
    import_id: Optional[int] = None
    last_action: Optional[str] = None
    task: int
    project: int
    updated_by: int
    parent_prediction: Optional[int] = None
    parent_annotation: Optional[int] = None
    last_created_by: Optional[int] = None


class TaskResultModel(BaseModel):
    id: int
    annotations: list[TaskAnnotationModel]
    meta: dict = Field()
    data: dict = Field(..., description="the task data")
    created_at: SerializableDatetimeAlways
    updated_at: SerializableDatetimeAlways
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Optional[SerializableDatetimeAlways] = None
    project: int
    updated_by: int
    comment_authors: list[int]

    @property
    def num_coders(self) -> int:
        return len(self.annotations)
