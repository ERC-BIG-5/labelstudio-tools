from typing import Optional

from ls_helper.annotation_timing import annotation_timing, plot_date_distribution
from ls_helper.annotations import _get_recent_annotations
from ls_helper.new_models import platforms_overview2
from main import open_image_simple


def status(p_access, accepted_ann_age: Optional[int] = 6):

    po = platforms_overview2.get_project(p_access)
    project_annotations = _get_recent_annotations(po.id, accepted_ann_age)

    df = annotation_timing(project_annotations)
    temp_file = plot_date_distribution(df)

    open_image_simple(temp_file.name)
    temp_file.close()

