from pathlib import Path

import numpy as np
import pandas as pd
from irrCAC.raw import CAC
from pandas import DataFrame

from ls_helper.new_models import ProjectResult


def get_default_df(df: DataFrame, question: str, default: str) -> DataFrame:
    # todo, this should use the reverse map , as we want to work with fixed names from here on

    # Get the valid task_id and ann_id combinations that exist in the original data
    valid_combinations = df[['task_id', 'ann_id', 'user_id']].drop_duplicates()

    # Create a complete DataFrame with valid combinations for the specific question
    complete_df = valid_combinations.copy()
    complete_df['question'] = question

    # Filter the original DataFrame for the specific question
    question_df = df[df["question"] == question].copy()

    # Then merge with question-specific data
    result = pd.merge(
        complete_df,
        question_df[['task_id', 'ann_id', "updated_at", 'question', 'value_idx', 'value']],
        on=['task_id', 'ann_id', 'question'],
        how='left'
    )

    # Fill missing values with default
    result['value'] = result['value'].fillna(default)
    result['value_idx'] = result['value_idx'].fillna(0).astype('int32')

    # Add any other columns needed from the original DataFrame
    if 'type' in df.columns:
        if len(question_df) > 0:
            result['type'] = question_df['type'].iloc[0]  # Use type from the question data
        else:
            type_col = df[df['question'] == question]['type'].iloc[0] if len(
                df[df['question'] == question]) > 0 else \
                df['type'].iloc[
                    0]
            result['type'] = type_col
    # todo verify
    if 'updated_at' in df.columns:
        if len(question_df) > 0:
            result['updated_at'] = question_df['updated_at'].iloc[0]  # Use type from the question data
        else:
            type_col = df[df['question'] == question]['updated_at'].iloc[0] if len(
                df[df['question'] == question]) > 0 else \
                df['updated_at'].iloc[
                    0]
            result['updated_at'] = type_col

    return result



def prep_single_select_agreement(df: DataFrame, question: str, indices: bool = True):
    """

    creates a pivot table, only containing the question results...
    :param mp:
    :param question:
    :return:
    """
    # aggfunc first?!?


    try:
        deff = get_default_df(df, question)
    except Exception as e:
        raise ValueError(f"err: {e}")

    pivot_df = deff.pivot_table(
        index='task_id',
        columns='user_id',
        values='value',
        aggfunc='first',  # Takes the first value if there are multiple,
        observed=False
    )
    if not indices:
        return pivot_df
    value_indices = [c.annot_val for c in mp.interface.orig_choices[question].options]

    deff = pivot_df.apply(
        lambda col: col.map(lambda x: value_indices.index(x) if (not pd.isna(x) and x in value_indices) else x))
    return deff


def check_all_empty(df: DataFrame) -> bool:
    return df.isna().all().all()

def calc_agreements(df: DataFrame):
    if check_all_empty(df):
        print("empty")
        return 1, 1
    cac_4raters = CAC(df)
    # print(cac_4raters)
    try:
        fleiss = cac_4raters.fleiss()
        gwet = cac_4raters.gwet()
        # print(f"{fleiss=} {gwet=}")
        return fleiss, gwet
    except Exception as e:
        print(e)
        return 1, 1


def calc_agreements2(deff: DataFrame
                     ):
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


def prep_multi_select_agreement(df, question, options) -> dict[str, DataFrame]:
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
    result_rows = {option: [] for option in options}

    # For each task_id and option combination
    for task_id in unique_task_ids:
        for option in options:
            # Create a base row with task_id and option
            row = {'task_id': task_id, 'option': option}

            # Add a column for each ann_id with 0 or 1
            for user_id in unique_user_ids:
                # Check if this option was selected for this task_id, ann_id
                selected = np.NaN
                annotations = df[(df['task_id'] == task_id) &
                                 (df['user_id'] == user_id) &
                                 (df['value_idx'] != -1)]

                if not annotations.empty and option in annotations['value'].values:
                    selected = 1
                # Add this ann_id as a column
                row[user_id] = selected

            result_rows[option].append(row)

    result_dfs = {}
    for option in options:
        result_dfs[option] = pd.DataFrame(result_rows[option]).set_index(['task_id']).drop("option", axis=1)

    """ all rows together
    all_rows = []
    for option in options:
        all_rows.extend(result_rows[option])
    result_dfs = DataFrame(all_rows).set_index(['task_id']).drop("option",axis=1)
    """

    return result_dfs


def export_annotations2csv(mp: ProjectResult) -> Path:
    df = mp.raw_annotation_df.copy()
    df = df.drop(["ann_id", "user", "updated_at"], axis=1)
    # drop the task_id,ann_id base rows
    df = df.dropna(subset=["type"])

    tasks_with_users = {}
    for task_id, group in df.groupby('task_id'):
        tasks_with_users[task_id] = group['user_id'].unique().tolist()

    # Now prepare the final dataframe
    result_rows = []

    # Process each task
    for task_id, users in tasks_with_users.items():
        row = {'task_id': task_id, 'user_ids': users}

        # Get all data for this task
        task_data = df[df['task_id'] == task_id]

        # Process each question that appears in this task
        for question in task_data['question'].dropna().unique():
            # Initialize a list of empty lists, one per user
            values_by_user = [[] for _ in users]

            # Get all answers for this question
            question_data = task_data[task_data['question'] == question]

            # Fill in values for each user
            for i, user_id in enumerate(users):
                user_values = question_data[question_data['user_id'] == user_id]['value'].dropna().tolist()
                values_by_user[i] = user_values

            # Add to row
            row[question] = values_by_user

        result_rows.append(row)

    # Create final dataframe
    final_result = pd.DataFrame(result_rows)

    ##
    def format_nested_lists(list_of_lists):
        if not isinstance(list_of_lists, list):
            return list_of_lists

        # Format each inner list as comma-separated values
        formatted_inner_lists = [','.join(map(str, inner_list)) for inner_list in list_of_lists]

        # Join the formatted inner lists with semicolons
        return ';'.join(formatted_inner_lists)

    final_result.to_csv("tt.csv")
