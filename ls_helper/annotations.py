import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from ls_helper.ana_res import parse_label_config_xml
from ls_helper.funcs import get_latest_annotation_file, get_latest_annotation
from ls_helper.models import ProjectAccess, MyProject, ProjectAnnotations, platforms_overview
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.new_models import platforms_overview2
from ls_helper.settings import ls_logger


def get_recent_annotations(project_id: int, accepted_age: int) -> Optional[ProjectAnnotations]:
    latest_file = get_latest_annotation_file(project_id)
    if latest_file is not None:
        file_dt = datetime.strptime(latest_file.stem, "%Y%m%d_%H%M")
        # print(file_dt, datetime.now(), datetime.now() - file_dt)
        if datetime.now() - file_dt < timedelta(hours=accepted_age):
            ls_logger.info("Get recent, gets latest annotation")
            return get_latest_annotation(project_id)

    print("downloading annotations")
    return ls_client().get_project_annotations(project_id)


def create_annotations_results(project: ProjectAccess, add_annotations: bool = True,
                               accepted_ann_age: Optional[int] = 6) -> MyProject:
    p_info = platforms_overview2.get_project(project)
    project_data = p_info.project_data()

    conf = p_info.get_structure()
    data_extensions = p_info.get_fixes()
    mp = MyProject(
        platform=p_info.platform,
        language=p_info.language,
        project_data=project_data,
        annotation_structure=conf,
        data_extensions=data_extensions)

    mp.apply_extension()
    if add_annotations:
        mp.raw_annotation_result = get_recent_annotations(p_info.id, accepted_ann_age)
        raw_annotation_df, assignment_df = mp.get_annotation_df()
        mp.raw_annotation_df = raw_annotation_df
        mp.assignment_df = assignment_df
    return mp



def convert_strings_to_indices(df: DataFrame, string_list: list[str]):
    result_df = df.copy()

    # Create a mapping dictionary for faster lookups
    # Each string maps to its index in the list
    string_to_index = {s: i for i, s in enumerate(string_list)}

    # Define a function to apply to each element
    def map_to_index(value):
        if pd.isna(value):
            return np.nan
        elif value in string_to_index:
            return string_to_index[value]
        else:
            # Optional: handle case when string is not in the list
            # Could return np.nan, -1, or raise an error
            return np.nan

    # Apply the function to all elements in the DataFrame
    result_df = result_df.map(lambda x: map_to_index(x))

    # Convert all columns to int32, with NaN represented as pd.NA
    for col in result_df.columns:
        # First convert to nullable integer type
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        result_df[col] = result_df[col].astype('Int32')  # Note: capital 'I' for nullable integer

    return result_df


def _reformat_for_datapipelines(mp, destinion_path: Path):
    """
    in order to assign assignment results to items in the sqlite database metadata-content
    :param mp:
    :return:
    """
    res = {}

    for task_result in mp.raw_annotation_result.annotations:
        res[task_result.data["platform_id"]] = task_result.model_dump(exclude={"data"})
    destinion_path.write_text(json.dumps(res))
    print(f"annotations reformatted -> {destinion_path.as_posix()}")
