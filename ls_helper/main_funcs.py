from typing import Optional

from ls_helper.annotation_timing import annotation_timing, plot_date_distribution
from ls_helper.new_models import platforms_overview
from ls_helper.project_mgmt import ProjectMgmt
from main import open_image_simple


def status(p_access, accepted_ann_age: Optional[int] = 6):

    po = platforms_overview.get_project(p_access)
    _, project_annotations = ProjectMgmt.get_recent_annotations(po.id, accepted_ann_age)

    df = annotation_timing(project_annotations)
    temp_file = plot_date_distribution(df)

    open_image_simple(temp_file.name)
    temp_file.close()

