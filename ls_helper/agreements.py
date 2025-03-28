import csv
import json
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from deprecated.classic import deprecated
from irrCAC.raw import CAC
from pandas import DataFrame
from pydantic import BaseModel

from ls_helper.annotations import create_df
from ls_helper.models import MyProject
from ls_helper.settings import SETTINGS


class AgreementsConfig(BaseModel):
    simple_columns: list[str]


default_agreement_columns = ["any_harmful", "nature_text", "nature_visual", "val-expr_text", "val-expr_visual",
                             "rel-value_text", "nep_materiality_text", "nep_biological_text", "landscape-type_text",
                             "basic-interaction_text", "stewardship_text", "disturbances_text",
                             "rel-value_visual", "nep_materiality_visual", "nep_biological_visual",
                             "landscape-type_visual",
                             "basic-interaction_visual", "stewardship_visual", "disturbances_visual"
                             ]


def calculate_jaccard(choices):
    """
    Calculate pairwise Jaccard similarity for multiple annotators' choices.

    Parameters:
    choices (list of lists): Each inner list contains choices selected by one annotator

    Returns:
    float: Average Jaccard similarity across all annotator pairs
    """
    if len(choices) < 2:
        return None

    similarities = []
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            set_i = set(choices[i])
            set_j = set(choices[j])

            if not set_i and not set_j:  # Both sets empty
                continue

            # Calculate Jaccard index: intersection size / union size
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))

            if union > 0:
                similarities.append(intersection / union)

    if similarities:
        return sum(similarities) / len(similarities)
    return None


@deprecated("use calc_agreements2")
def calc_agreements(
        mp: MyProject,
        min_num_coders: int,
        agreement_columns: Optional[list[str]] = None) -> tuple[Path, Path]:
    if not agreement_columns:
        agreement_columns = default_agreement_columns

    all_rel = []

    struct_ch = mp.annotation_structure.choices
    csv_filepath = SETTINGS.agreements_dir / f"agreements_{mp.platform}_{mp.language}_{mp.raw_annotation_result.file_path.stem}.csv"
    pid_filepath = SETTINGS.agreements_dir / f"platform_ids_{mp.platform}_{mp.language}_{mp.raw_annotation_result.file_path.stem}.json"
    fieldnames = ["column", "choice_type", "coefficient", "filtered_count", "positive_match", "filtered_coefficient",
                  "jaccard_index"]

    options_dict = {}

    def _choice_type(col) -> str:
        return "single" if struct_ch[col].choice == "single" else "multiple"

    # print(len(results.annotation_results))
    # for each col (final col), store the platform_ids of filtered/positive/conflict(=filtered-positive)
    platform_ids = []
    for task in mp.annotation_results:
        if task.num_coders < min_num_coders:
            continue
        platform_ids.append(task.relevant_input_data["platform_id"])
        res = task.data()
        # check if we skip skippers
        vals = {c: (res.get(c) or [])[:min_num_coders] for c in agreement_columns}

        for col in agreement_columns:
            if col not in options_dict:
                options = [c.alias if c.alias else c.value
                           for c in mp.annotation_structure.choices[col].options]
                orig_name = mp.data_extensions.fix_reverse_map[col]
                default = mp.data_extensions.fixes[orig_name].default
                if default and default not in options:
                    options.append(default)
                options_dict[col] = options
            else:
                options = options_dict[col]

            res = vals[col]
            for add in range(min_num_coders - len(res)):
                res.append([])
            del vals[col]
            for o in options:
                vals[f"{col}-[{o}]"] = []
                # For each coder's results
                # 1 if option is in coder's choices, 0 otherwise
                vals[f"{col}-[{o}]"] = [1 if o in coder_choices else 0 for coder_choices in res]

        all_rel.append(vals)

    fout = csv_filepath.open('w', newline='')
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    pids = {}

    for col in agreement_columns:
        # print(col)
        cols_names = []

        option_cols = []
        options = options_dict[col]
        # Get all the binary columns for this base column
        for o in options:
            option_col_name = f"{col}-[{o}]"
            cols_names.append(option_col_name)
            option_col_data = [r.get(option_col_name, 0) for r in all_rel]
            option_cols.append(option_col_data)

        cols = option_cols

        for col_name, col_ in zip(cols_names, cols):
            # print(col_name, len(col_))
            if all(not d for d in col_):
                print(f"No data for : {col}")
                continue

            choice_type = _choice_type(col)

            csv_row = {
                "column": col_name,
                "choice_type": choice_type,
                "coefficient": None,
                "filtered_count": 0,
                "positive_match": 0,
                "filtered_coefficient": None,
                "jaccard_index": None
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

                df = pd.DataFrame(col_)
                cac_4raters = CAC(df)
                gwet_res = cac_4raters.gwet()
                csv_row["coefficient"] = gwet_res["est"]["coefficient_value"]

                # if choice_type != "single":
                # Filter rows to include only those where at least one coder selected this option
                filtered_mask = [any(row) for row in col_]
                positive_mask = [all(row) for row in col_]
                conflicting_mask = [any(row) and not all(row) for row in col_]
                filtered_col_data = [row for row, mask in zip(col_, filtered_mask) if mask]

                if filtered_col_data:  # Only proceed if there's data after filtering
                    filtered_df = pd.DataFrame(filtered_col_data)
                    num_filtered = len(filtered_df)
                    csv_row["filtered_count"] = num_filtered
                    # print(f"Filtered {num_filtered} rows")
                    if num_filtered > 1:
                        # Compute agreement on filtered data
                        # print(filtered_df)
                        filtered_cac = CAC(filtered_df)
                        filtered_gwet = filtered_cac.gwet()
                        both_selected = [row for row in col_ if all(row)]
                        csv_row["positive_match"] = len(both_selected)
                        csv_row["filtered_coefficient"] = filtered_gwet["est"]["coefficient_value"]
                        # print(f"Positive match: {len(both_selected)}")
                        # print(f"Filtered coefficient: {filtered_gwet['est']['coefficient_value']}")

                        # jaccard_similarities = []
                        original_choices = []

                        # Extract original choices for each annotator
                        for task_data in all_rel:
                            annotator_choices = []
                            for o in options:
                                option_col_name = f"{col}-[{o}]"
                                if task_data.get(option_col_name, 0) == 1:
                                    annotator_choices.append(o)
                            original_choices.append(annotator_choices)

                        # Calculate Jaccard for filtered data only
                        filtered_choices = [choices for choices, mask in zip(original_choices, filtered_mask) if
                                            mask]
                        jaccard_score = calculate_jaccard(filtered_choices)

                        # Add to CSV row
                        csv_row["jaccard_index"] = jaccard_score

                    elif num_filtered == 1:
                        csv_row["positive_match"] = len([row for row in col_ if all(row)])
                        # print(f"Positive match: {len([row for row in col_ if all(row)])}")
                        # print(filtered_df[0])
                    else:
                        pass
                        # print(f"No filtered coefficient")

                # Write platform IDs for this column to the separate CSV
                id_row = {
                    "filtered_ids": [pid for pid, mask in zip(platform_ids, filtered_mask) if mask],
                    "positive_ids": [pid for pid, mask in zip(platform_ids, positive_mask) if mask],
                    "conflicting_ids": [pid for pid, mask in zip(platform_ids, conflicting_mask) if mask]
                }
                pids[col_name] = id_row

            # print(gwet_res)
            # print(gwet_res["est"]["coefficient_value"])
            writer.writerow(csv_row)

    pid_filepath.write_text(json.dumps(pids))
    print(f"Multi-choices task platform-ids -> {pid_filepath.as_posix()}")

    fout.close()
    print(f"agreements -> {csv_filepath.as_posix()}")
    return csv_filepath, pid_filepath


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


def prepare_df(df: DataFrame, use_value_indices: bool = True,
               include_questions: set[str] = None,
               multi_select_options: dict[str, list[str]] = None) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
    """
    Prepare pandas DataFrame for irrCAC analysis.

    Parameters:
    df (pandas.DataFrame): DataFrame with columns: task_id, item_id, coder_id, 
                          question_id, question_type, response

    Returns:
    dict: Dictionary of DataFrames ready for irrCAC, with item_id as keys
    """

    # Filter for only the questions you want to include
    if include_questions is not None:
        base_df = df[df['question_id'].isin(list(include_questions))].copy()
    else:
        base_df = df.copy()

    single_type_df = prepare_single_select_for_irrcac(base_df)
    multiple_type_df = prepare_multiselect_for_irrcac(base_df, multi_select_options)
    return single_type_df, multiple_type_df


@deprecated("")
def calc_agreements2(
        res: MyProject,
        selected_categories: Sequence[str]
):
    df = create_df(res)

    multi_choice_options = {}

    rows = []

    for c_name, c in res.annotation_structure.choices.items():
        if c.choice == "multiple":
            multi_choice_options[c_name] = c.indices

    single_select, multiple_select = prepare_df(df, True, set(selected_categories), multi_choice_options)

    for cat in selected_categories:
        if cat not in single_select:
            continue
        print(cat, "single")
        try:
            cac_4raters = CAC(single_select[cat])
            # print(cac_4raters)
            fleiss = cac_4raters.fleiss()["est"]["coefficient_value"]
            gwet = cac_4raters.gwet()["est"]["coefficient_value"]
            print(f"{fleiss=} {gwet=}")
            rows.append(
                {"question": cat, "choice-type": "single", "coefficient-kappa": fleiss, "coefficient-gwet": gwet})
        except Exception as e:
            print(e)

    for cat in selected_categories:
        if cat not in multiple_select:
            continue
        m_df = multiple_select[cat]
        agreement_dfs = {val: group for val, group in m_df.groupby('response_idx')}
        print(cat, "multiple")
        for val, a in agreement_dfs.items():
            print(multi_choice_options[cat][val])
            is_all_zeros = (a == 0).all().all()
            if is_all_zeros:
                print("None")
                continue
            cac_4raters = CAC(a)
            fleiss = cac_4raters.fleiss()["est"]["coefficient_value"]
            gwet = cac_4raters.gwet()["est"]["coefficient_value"]
            print(f"{fleiss=} {gwet=}")
            rows.append(
                {"question": f"{cat}-[{multi_choice_options[cat][val]}]", "choice-type": "multiple",
                 "coefficient-kappa": fleiss, "coefficient-gwet": gwet})

    DataFrame(rows).to_csv(Path("agreements.csv"))


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
