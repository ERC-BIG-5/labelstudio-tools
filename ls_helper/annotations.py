from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
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
        mp.annotation_structure.apply_extension(mp.data_extensions)
        mp.calculate_results()
    return mp


def create_df(mp: MyProject) -> DataFrame:
    task_results = mp.annotation_results
    # Create lists to store our data
    rows = []

    # For each task result object
    for task_idx, task_result in enumerate(task_results):
        # For each item in the task
        for item_id, item in task_result.items.items():
            # Skip anything that's not single or multiple type
            if item.type_ not in ["single", "multiple"]:
                continue

            # For each coder
            for user_idx, user_id in enumerate(item.users):
                # Get this user's values for this item
                if user_idx < len(item.values):
                    user_values,user_values_idx = item.values[user_idx], item.value_indices[user_idx]

                    for val,val_idx in zip(user_values, user_values_idx):
                    # For single choice, just one value per user per item
                        rows.append({
                            "task_id": task_result.task_id,
                            "task_idx": task_idx,
                            "item_id": item_id,
                            "coder_id": user_id,
                            "question_id": item.name,
                            "question_type": item.type_,
                            "response": val,
                            "response_idx": val_idx
                        })


    # Create the DataFrame
    df = pd.DataFrame(rows)
    return df


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


def annotations_calculations2(p_info: MyProject):
    pass

if __name__ == "__main__":
    pass

    """ merge to create unifixes DONE
    merge = {}
    print(yt_t -  twitter_f)
    print(twitter_f - yt_t)
    i_s = yt_t.intersection(twitter_f)
    print(i_s)
    for k in i_s:
        print(k)
        if yt.fixes[k].name_fix != twitter.fixes[k].name_fix:
            print(k, yt.fixes[k].name_fix,twitter.fixes[k].name_fix )

        merge[k] = yt.fixes[k]

    merge = {k: merge[k] for k in yt.fixes.keys() if k in i_s}

    #SETTINGS.unifix_file_path.write_text(ProjectAnnotationExtension(project_id=0, fixes=merge).model_dump_json(indent=2))
    """

    """ removed the fixes in the platforms which are in the unifix DONE
    print("tw")
    for k in [k for k in twitter.fixes.keys() if k in uni_f]:
        print(k)
        del twitter.fixes[k]

    (SETTINGS.fixes_dir /  "39.json").write_text(twitter.model_dump_json(indent=2, exclude_none=True))
    print("-------")
    print("yt")
    for k in [k for k in yt.fixes.keys() if k in uni_f]:
        print(k)
        del yt.fixes[k]

    #(SETTINGS.fixes_dir / "33.json").write_text(yt.model_dump_json(indent=2, exclude_none=True))
    """
