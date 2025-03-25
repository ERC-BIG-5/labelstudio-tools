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


def multi_choice_rows(mp: MyProject, question: str):
    options = mp.annotation_structure.choices.get(question).raw_options_list()


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
    # Get all unique index combinations
    idx_combinations = df[['task_id', 'ann_id']].drop_duplicates()

    # Filter the dataframe for the specific question
    question_df = df[df['question'] == question].copy()

    if question_df.empty:
        result_df = pd.DataFrame(index=pd.MultiIndex.from_frame(idx_combinations))
    else:
        # Create indicator columns using get_dummies and add 'x_' prefix
        # Only create dummies if we have values
        dummies = pd.get_dummies(question_df['value'], prefix='x')

        # Combine with the relevant index columns
        result_df = pd.concat([question_df[['task_id', 'ann_id']], dummies], axis=1)

        # Group by the indices and take the maximum (this handles duplicates)
        result_df = result_df.groupby(['task_id', 'ann_id']).max()

    # Create a complete index from all combinations
    full_idx = pd.MultiIndex.from_frame(idx_combinations)

    # Reindex with all possible index combinations
    result_df = result_df.reindex(full_idx)

    # Handle the boolean/NaN conversion properly:
    # First convert to float (which can handle NaN), then fill NaN, then convert to int
    result_df = result_df.astype(float).fillna(0).astype(int)

    # Reset index
    result_df = result_df.reset_index()

    # Define expected column names based on options
    expected_cols = [f"x_{val}" for val in options]#[f'{question}_[{val}]' for val in options]

    # Add missing columns if needed
    for col in expected_cols:
        if col not in result_df.columns:
            result_df[col] = 0

    # Keep only the columns we need
    # First ensure we have task_id and ann_id
    result_cols = ['task_id', 'ann_id']

    # Then add any expected columns that exist
    for col in expected_cols:
        result_cols.append(col)

    # Select only the columns that actually exist in the dataframe
    available_cols = [col for col in result_cols if col in result_df.columns]
    result_df = result_df[available_cols]

    # Add any missing expected columns with zeros
    for col in expected_cols:
        if col not in result_df.columns:
            result_df[col] = 0

    return result_df
