from typing import Optional

from ls_helper.annotations import get_recent_annotations
from ls_helper.new_models import platforms_overview2


def status(platform: str, language: str, name: Optional[str] = None, accepted_ann_age: Optional[int] = 6):
    po = platforms_overview2.get_project((platform, language))
    project_annotations = get_recent_annotations(po.id, accepted_ann_age)

    df = annotation_timing(project_annotations)
    temp_file = plot_date_distribution(df)

    open_image_simple(temp_file.name)
    temp_file.close()
