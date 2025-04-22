from typing import Optional

from ls_helper.annotation_timing import (
    annotation_timing,
    plot_date_distribution,
)
from ls_helper.new_models import (
    ProjectAnnotationResultsModel,
    ProjectData,
)
from ls_helper.project_mgmt import ProjectMgmt
from main import open_image_simple


def status(project: ProjectData, accepted_ann_age: int = 6):
    _, project_annotations = ProjectMgmt.get_recent_annotations(
        project.id, accepted_ann_age
    )
    pa: Optional[ProjectAnnotationResultsModel] = project_annotations
    if project_annotations:
        df = annotation_timing(pa, project.project_data.maximum_annotations)
        temp_file = plot_date_distribution(df)
        open_image_simple(temp_file.name)
        temp_file.close()
