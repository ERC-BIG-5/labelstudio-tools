from datetime import datetime
from typing import Optional, Any, TypedDict, Literal, Annotated

from pydantic import BaseModel, Field, PlainSerializer


class ProjectModel(BaseModel):
    id: int
    title: str
    description: str
    label_config: str
    expert_instruction: str
    show_instruction: bool

    show_skip_button: Optional[bool] = Field(default=None)
    """
    Show a skip button in interface and allow annotators to skip the task
    """

    enable_empty_annotation: Optional[bool] = Field(default=None)
    """
    Allow annotators to submit empty annotations
    """

    show_annotation_history: Optional[bool] = Field(default=None)
    """
    Show annotation history to annotator
    """

    organization: Optional[int] = None
    color: Optional[str] = None
    maximum_annotations: Optional[int] = Field(default=None)
    """
    Maximum number of annotations for one task. If the number of annotations per task is equal or greater to this value, the task is completed (is_labeled=True)
    """

    is_published: Optional[bool] = Field(default=None)
    """
    Whether or not the project is published to annotators
    """

    model_version: Optional[str] = Field(default=None)
    """
    Machine learning model version
    """

    is_draft: Optional[bool] = Field(default=None)
    """
    Whether or not the project is in the middle of being created
    """

    # TODO
    created_by: Optional[dict] = None  # UserSimple
    created_at: Optional[datetime] = None
    min_annotations_to_start_training: Optional[int] = Field(default=None)
    """
    Minimum number of completed tasks after which model training is started
    """

    start_training_on_annotation_update: Optional[str | bool] = Field(default=None)
    """
    Start model training after any annotations are submitted or updated
    """

    show_collab_predictions: Optional[bool] = Field(default=None)
    """
    If set, the annotator can view model predictions
    """

    num_tasks_with_annotations: Optional[int] = Field(default=None)
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
    sampling: Optional[dict | str] = None  # ProjectSampling
    show_ground_truth_first: Optional[bool] = None
    show_overlap_first: Optional[bool] = None
    overlap_cohort_percentage: Optional[int] = None
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

    evaluate_predictions_automatically: Optional[bool] = Field(default=None)
    """
    Retrieve and display predictions when loading a task
    """

    config_has_control_tags: Optional[str | bool] = Field(default=None)
    """
    Flag to detect is project ready for labeling
    """
    # todo
    skip_queue: Optional[dict | str] = None  # ProjectSkipQueue
    reveal_preannotations_interactively: Optional[bool] = Field(default=None)
    """
    Reveal pre-annotations interactively
    """

    pinned_at: Optional[datetime] = Field(default=None)
    """
    Pinned date and time
    """

    finished_task_number: Optional[int] = Field(default=None)
    pass


# from LS
class ProjectViewsDataModel(BaseModel):
    type: Optional[str] = None
    title: str
    target: Optional[str] = None
    gridWidth: Optional[int] = None
    columnWidth: Optional[int] = None
    hiddenColumns: Optional[
        TypedDict("hiddenColumns", {"explore": list["str"], "labeling": Optional[list[str]]})] = None
    semantic_search: Optional[list] = None
    columnsDisplay: Optional[dict] = None
    filters: TypedDict("filters", {"conjunction": Literal["and", "or"], "items": list[TypedDict("items", {
        "filter": str, "operator": str, "type": str, "value": str | int
    })]})
    ordering: Optional[list[str]] = None


# from LS
class ProjectViewModel(BaseModel):
    id: int
    order: int
    user: int
    project: int
    data: ProjectViewsDataModel


class UserModel(BaseModel):
    id: int
    first_name: str
    last_name: str
    username: str
    email: str
    last_activity: Annotated[datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str, when_used='always')]
    avatar: Optional[str] = None
    initials: str
    phone: Optional[str] = None
    active_organization: int
    allow_newsletters: Optional[bool] = False
    # date_joined: datetime
