import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from deprecated.classic import deprecated
from pandas import DataFrame

from ls_helper.ana_res import parse_label_config_xml
from ls_helper.funcs import get_latest_annotation_file, get_latest_annotation
from ls_helper.models import ProjectAccess, MyProject, ProjectAnnotations, platforms_overview
from ls_helper.my_labelstudio_client.client import ls_client
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
    p_info = platforms_overview.get_project(project)
    project_data = p_info.project_data()

    conf = parse_label_config_xml(project_data.label_config)

    data_extensions = p_info.get_fixes()
    mp = MyProject(
        platform=p_info.platform,
        language=p_info.language,
        project_data=project_data,
        annotation_structure=conf,
        data_extensions=data_extensions)

    if add_annotations:
        mp.raw_annotation_result = get_recent_annotations(p_info.id, accepted_ann_age)
        mp.apply_extension()
        mp.get_annotation_df()
    return mp


@deprecated("straight to MyProject.get_annotation_df")
def create_df(mp: MyProject) -> DataFrame:
    task_results = mp.annotation_results
    # Create lists to store our data
    rows = []

    # For each task result object
    for task_idx, task_result in enumerate(task_results):
        # print(task_result.task_id)
        # For each item in the task
        for item_id, item in task_result.items.items():
            # Skip anything that's not single or multiple type

            if item.type_ not in ["single", "multiple", "list-single", "list-multiple"]:
                # text
                # print(item.type_)
                continue

            # For each coder
            for user_idx, user_id in enumerate(item.users):
                # Get this user's values for this item
                if user_idx < len(item.values):
                    user_values, user_values_idx = item.values[user_idx], item.value_indices[user_idx]

                    for idx, (val, val_idx) in enumerate(zip(user_values, user_values_idx)):
                        # For single choice, just one value per user per item
                        if item.type_.startswith("list"):
                            if not val:
                                continue
                            for i_idx, i_val in enumerate(val):
                                rows.append({
                                    "task_id": task_result.task_id,
                                    "task_idx": task_idx,
                                    "coder_id": user_id,
                                    "question_id": item.name,
                                    "question_type": item.type_,
                                    "response": i_val,
                                    "list_index": idx,
                                    "response_idx": val_idx[i_idx] if val_idx else None  # could be text
                                })
                        else:
                            rows.append({
                                "task_id": task_result.task_id,
                                "task_idx": task_idx,
                                "coder_id": user_id,
                                "question_id": item.name,
                                "question_type": item.type_,
                                "response": val,
                                "list_index": 0,
                                "response_idx": val_idx
                            })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    return df


def prepare_for_question(mp, question):
    """
    not sure, if that works well...
    :param mp:
    :param question:
    :return:
    """
    df = mp.raw_annotation_df
    # Get the valid task_id and ann_id combinations that exist in the original data
    valid_combinations = df[['task_id', 'ann_id', 'user_id']].drop_duplicates()

    # Create a complete DataFrame with valid combinations for the specific question
    complete_df = valid_combinations.copy()
    complete_df['question'] = question

    # Filter the original DataFrame for the specific question
    question_df = df[df["question"] == question].copy()
    return question_df


@deprecated("not sure what this does")
def prepare_numeric_agreement(df):
    # Create a mapping of category strings to numeric values for each question
    category_maps = {}

    for question in df['question_id'].unique():
        q_df = df[df['question_id'] == question]
        categories = sorted(q_df['response'].dropna().unique())  # Sort and drop NAs
        category_maps[question] = {cat: i for i, cat in enumerate(categories)}

    # Process each question separately to maintain integer types
    pivot_dfs = {}

    for question in df['question_id'].unique():
        # Filter data for this question (single-choice only)
        q_df = df[(df['question_id'] == question) & (df['question_type'] == 'single')].copy()

        # Create numeric responses using the mapping
        q_df['response_numeric'] = q_df['response'].map(lambda x:
                                                        category_maps[question].get(x, pd.NA) if pd.notna(x) else pd.NA)

        # Convert to integer explicitly before pivoting
        q_df['response_numeric'] = pd.to_numeric(q_df['response_numeric'],
                                                 downcast='integer',
                                                 errors='coerce')

        # Create the pivot table for this question
        pivot = q_df.pivot_table(
            index=['task_id', 'item_id'],
            columns='coder_id',
            values='response_numeric',
            aggfunc='first'  # In case of duplicates
        )

        # Force integer dtype after pivot (convert back from float)
        for col in pivot.columns:
            # Check if column can be safely converted to integer
            if pivot[col].notna().all():
                pivot[col] = pivot[col].astype(int)

        pivot_dfs[question] = pivot.fillna(-1).astype(int)

    return pivot_dfs, category_maps


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
