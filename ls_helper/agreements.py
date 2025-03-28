import re

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel



class AgreementsConfig(BaseModel):
    simple_columns: list[str]


default_agreement_columns = ["any_harmful", "nature_text", "nature_visual", "val-expr_text", "val-expr_visual",
                             "rel-value_text", "nep_materiality_text", "nep_biological_text", "landscape-type_text",
                             "basic-interaction_text", "stewardship_text", "disturbances_text",
                             "rel-value_visual", "nep_materiality_visual", "nep_biological_visual",
                             "landscape-type_visual",
                             "basic-interaction_visual", "stewardship_visual", "disturbances_visual"
                             ]



def prepare_single_select_for_irrcac(base_df: DataFrame, use_value_indices: bool = True) -> dict[str, DataFrame]:
    # Filter for only 'single' question types
    single_type_df = base_df[base_df['question_type'].isin(['single', "list-single"])].copy()
    # Get unique item_ids, coder_ids, and task_ids
    item_ids = single_type_df['question_id'].unique()

    # Prepare a dictionary to hold DataFrames for each item_id
    irrcac_data = {}

    # For each item_id, create a pivot table
    for item_id in item_ids:
        item_df = single_type_df[single_type_df['question_id'] == item_id].copy()

        # Create pivot table: rows=task_id, columns=coder_id, values=response
        pivot_df = item_df.pivot_table(
            index=['task_id', 'task_idx', 'list_index'],
            columns='coder_id',
            values='response_idx',
            aggfunc='first'  # In case of duplicates, take the first response
        )

        # Store in our dictionary
        if use_value_indices:
            irrcac_data[item_id] = pivot_df.astype('Int64')
        else:
            irrcac_data[item_id] = pivot_df

    return irrcac_data


def prepare_multiselect_for_irrcac(df,
                                   options_dict=None):
    """
    Transform multi-select questions into a format matching the single-select structure,
    with one DataFrame per item_id (question), and response_idx as another level in the index.

    Parameters:
    df (pandas.DataFrame): DataFrame with columns: task_id, item_id, coder_id, question_id,
                          question_type, response, response_idx
    options_dict (dict, optional): Dictionary mapping from item_id to list of possible options

    Returns:
    dict: Dictionary with item_id keys and DataFrames as values
    """
    # Filter for only 'multiple' question types
    multiple_type_df = df[df['question_type'].isin(['multiple', "list-multiple"])].copy()

    # Get unique item_ids
    questions = multiple_type_df['question_id'].unique()

    # Dictionary to store results
    result_dict = {}

    # Process each item_id (question) separately
    for item_id in questions:
        item_df = multiple_type_df[multiple_type_df['question_id'] == item_id].copy()

        # Get all possible options for this item_id
        if options_dict and item_id in options_dict:
            options = options_dict[item_id]
        else:
            # If no options provided, use all unique responses
            options = item_df['response'].unique().tolist()

        # Get unique task combinations
        task_combos = item_df[['task_id', 'task_idx', 'list_index']].drop_duplicates()

        # Create a list to store rows
        rows = []

        # For each task
        for _, task_row in task_combos.iterrows():
            task_id = task_row['task_id']
            task_idx = task_row['task_idx']
            list_idx = task_row['list_index']

            # For each option (response_idx)
            for resp_idx, option in enumerate(options):
                # Create a row dictionary with task info and response_idx
                row_dict = {
                    'task_id': task_id,
                    'task_idx': task_idx,
                    'list_index': list_idx,
                    "response": options[resp_idx],
                    'response_idx': resp_idx
                }

                # For each coder, add whether they selected this option
                for coder_id in item_df['coder_id'].unique():
                    coder_responses = item_df[
                        (item_df['task_id'] == task_id) &
                        (item_df['task_idx'] == task_idx) &
                        (item_df['list_index'] == list_idx) &
                        (item_df['coder_id'] == coder_id)
                        ]['response'].tolist()

                    row_dict[coder_id] = 1 if option in coder_responses else 0

                # Add to rows list
                rows.append(row_dict)

        # Convert to DataFrame
        result_df = pd.DataFrame(rows)
        # Set all columns except coder IDs as index
        index_cols = ['task_id', 'task_idx', 'list_index', 'response', 'response_idx']
        result_df = result_df.set_index(index_cols)

        # Store in result dictionary
        result_dict[item_id] = result_df

    return result_dict


def add_image_index_colum(df: DataFrame):
    def extract_image_indices(col_text: str):
        if pd.isna(col_text):
            return 0, "NaN"
        # merge all individual indices in the naming ..._ID_... into a one level deper nesting.
        pattern = r'_(\d+)_|_(\d+)$'
        match = re.search(pattern, col_text)
        if match:
            # #todo use the number to order, guarantee right order
            number = int(match.group(0).strip("_"))
            group_name = re.sub(pattern, "_#_", col_text).strip("_")
            print(number, group_name)
            return number, group_name
        else:
            return 0, col_text

    index = []
    group_names = []

    for value in df["question"]:
        num, group = extract_image_indices(value)
        index.append(num)
        group_names.append(group)

    image_index_results = pd.DataFrame({
        'index': index,
        'group_name': group_names
    })

    df["question"] = image_index_results["group_name"]
    df["image_idx"] = image_index_results["index"]


# Example usage
# df = pd.read_csv('your_data.csv')
# irrcac_data = prepare_data_for_irrcac(df)

# To use with irrCAC:
# for item_id, data in irrcac_data.items():
#     # Extract just the coder columns (removing task_id and task_idx)
#     coder_data = data.iloc[:, 2:]
#
#     # Run CAC analysis
#     result = CAC(coder_data)
#     print(f"Results for {item_id}:")
#     print(result)
