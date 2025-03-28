import numpy as np
import pandas as pd
from irrCAC.raw import CAC
from pandas import DataFrame

from ls_helper.models import MyProject


def prepare_df_for_agreement(mp: MyProject, question: str, indices: bool = True):
    """
    creates a pivot table, only containing the question results...
    :param mp:
    :param question:
    :return:
    """
    # aggfunc first?!?
    try:
        deff = mp.get_default_df(question)
    except Exception as e:
        print(f"err: {e}")
        deff = mp.raw_annotation_df
    pivot_df = deff.pivot_table(
        index='task_id',
        columns='user_id',
        values='value',
        aggfunc='first',  # Takes the first value if there are multiple,
        observed=False
    )
    if not indices:
        return pivot_df
    value_indices = [c.annot_val for c in mp.annotation_structure.choices[question].options]

    deff = pivot_df.apply(
        lambda col: col.map(lambda x: value_indices.index(x) if (not pd.isna(x) and x in value_indices) else x))
    return deff


def calc_agreements2(deff: DataFrame):
    cac_4raters = CAC(deff)
    # print(cac_4raters)
    try:
        fleiss = cac_4raters.fleiss()["est"]["coefficient_value"]
        gwet = cac_4raters.gwet()["est"]["coefficient_value"]
        # print(f"{fleiss=} {gwet=}")
        return fleiss, gwet
    except Exception as e:
        print(e)
        return 1, 1


def prep_multi_select(df, question, options):
    """
    Create a binary indicator DataFrame for multi-select questions.

    Parameters:
    - df: DataFrame containing the data
    - question: The specific question to filter by
    - options: List of expected option values

    Returns:
    - DataFrame with task_id, ann_id, and binary indicators for each option
    """

    df = df[df["question"].isin([question, np.NaN])]
    df = df.drop(["user", "updated_at"], axis=1)

    # Get unique task_ids and ann_ids
    unique_task_ids = df['task_id'].unique()
    unique_user_ids = df['user_id'].unique()

    # Create a result DataFrame with task_id and option as indices
    result_rows = []

    # For each task_id and option combination
    for task_id in unique_task_ids:
        for option in options:
            # Create a base row with task_id and option
            row = {'task_id': task_id, 'option': option}

            # Add a column for each ann_id with 0 or 1
            for user_id in unique_user_ids:
                # Check if this option was selected for this task_id, ann_id
                selected = 0
                annotations = df[(df['task_id'] == task_id) &
                                 (df['user_id'] == user_id) &
                                 (df['value_idx'] != -1)]

                if not annotations.empty and option in annotations['value'].values:
                    selected = 1

                # Add this ann_id as a column
                row[user_id] = selected

            result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    return result_df
