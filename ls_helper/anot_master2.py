from typing import Optional

import numpy as np
import pandas as pd
import re
from irrCAC.raw import CAC
from numpy import isnan
from pandas import DataFrame

from ls_helper.models.field_models import FieldModel, FieldType, ChoiceFieldModel, NO_SINGLE_CHOICE
from tools.project_logging import get_logger

logger = get_logger(__file__)


def identify_conflicts(agreement_matrix: DataFrame, variable: str, variable_info: ChoiceFieldModel,
                       variable_annotations):
    """
    Identify conflicts for this variable.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    variable : str
        ID of the variable

    variable_info : ChoiceFieldModel
        Variable metadata including type

    variable_annotations : DataFrame
        Annotations for this variable with all metadata

    Returns:
    --------
    list of conflict dictionaries
    """
    conflicts = []

    # Get a mapping of task_id to platform_id from the annotations
    platform_id_map = {}
    for _, row in variable_annotations.iterrows():
        if 'task_id' in row and 'platform_id' in row:
            platform_id_map[row['task_id']] = row['platform_id']

    # Check if we're using composite keys (for indexed variables)
    has_composite_key = False
    if len(agreement_matrix.index) > 0:
        first_idx = agreement_matrix.index[0]
        has_composite_key = isinstance(first_idx, str) and '_' in first_idx

    # Different handling for single vs multiple choice variables
    if variable_info.choice == 'single':
        # For single-select, identify conflicts where annotators chose different options
        for task_id, row in agreement_matrix.iterrows():
            # Drop NaN values and get the values and corresponding coders
            values = row.dropna().tolist()
            coders = row.dropna().index.tolist()

            if len(values) < 2:
                continue  # Need at least 2 annotations to have a conflict

            # For single-choice, extract the first element from any lists
            processed_values = []
            for val in values:
                if isinstance(val, list) and len(val) > 0:
                    processed_values.append(val[0])
                else:
                    processed_values.append(val)

            # Check if all values are equal
            unique_values = set()
            for val in processed_values:
                if isinstance(val, (int, float, str)):
                    unique_values.add(val)

            if len(unique_values) <= 1:
                continue  # All values are the same, no conflict

            # Extract actual task_id and image_idx if using composite keys
            actual_task_id = task_id
            image_idx = None

            if has_composite_key:
                parts = str(task_id).split('_')
                if len(parts) >= 2:
                    actual_task_id = parts[0]
                    try:
                        image_idx = int(parts[1])
                    except (ValueError, IndexError):
                        pass

            # Find platform_id for this task
            platform_id = platform_id_map.get(actual_task_id)

            # Convert numeric indices to option labels for readability
            labeled_annotations = []
            for i, value in enumerate(processed_values):
                if isinstance(value, (int, float)) and not np.isnan(value):
                    try:
                        # Convert index to label using the options list
                        label = variable_info.options[int(value)]
                    except (IndexError, ValueError):
                        label = str(value) + " (unknown option)"
                else:
                    label = str(value)

                labeled_annotations.append({
                    'user_id': coders[i],
                    'value': label
                })

            # Record conflict
            conflict = {
                'task_id': actual_task_id,
                'platform_id': platform_id,
                'variable': variable,
                'agreement_score': 0.0,  # 0 for a conflict
                'annotations': labeled_annotations
            }

            # Add image_idx if available
            if image_idx is not None:
                conflict['image_idx'] = image_idx

            conflicts.append(conflict)

    else:  # Multiple choice
        # For multi-select, we need to check conflicts option by option
        options = variable_info.options

        for option in options:
            # Create binary matrix for this option (1 if present, 0 if absent)
            binary_matrix = agreement_matrix.applymap(
                lambda x: 1 if isinstance(x, list) and option in x else 0
            )

            # Look for disagreements on this option
            for task_id, row in binary_matrix.iterrows():
                # Drop NaN values
                values = row.dropna()
                if len(values) < 2:
                    continue  # Need at least 2 annotations

                # Check if there's disagreement (mix of 0s and 1s)
                if values.nunique() <= 1:
                    continue  # All annotators agree on this option

                # Extract actual task_id and image_idx if using composite keys
                actual_task_id = task_id
                image_idx = None

                if has_composite_key:
                    parts = str(task_id).split('_')
                    if len(parts) >= 2:
                        actual_task_id = parts[0]
                        try:
                            image_idx = int(parts[1])
                        except (ValueError, IndexError):
                            pass

                # Create annotations showing which users marked this option as present/absent
                option_annotations = []
                for coder, value in values.items():
                    option_annotations.append({
                        'user_id': coder,
                        'option': option,
                        'value': 'present' if value == 1 else 'absent'
                    })

                # Record conflict for this option
                conflict = {
                    'task_id': actual_task_id,
                    'platform_id': platform_id_map.get(actual_task_id),
                    'variable': variable,
                    'option': option,  # Specify which option has the conflict
                    'agreement_score': 0.0,
                    'annotations': option_annotations,
                    'conflict_type': 'multiple_choice'
                }

                # Add image_idx if available
                if image_idx is not None:
                    conflict['image_idx'] = image_idx

                conflicts.append(conflict)

    return conflicts

def analyze_coder_agreement(raw_annotations, assignments, choices: dict[str, ChoiceFieldModel],
                            field_names: Optional[list[str]] = None) -> dict:
    """
    End-to-end function to analyze coder agreement across annotations.

    Parameters:
    -----------
    raw_annotations : list of dict or DataFrame
        List of annotation objects with keys:
        task_id, ann_id, coder_id, ts, platform_id, category (variable name), type, value

    assignments : list of dict or DataFrame
        List of task-coder assignments with metadata

    choices : dict
        Dictionary mapping variable_ids to their ChoiceFieldModel objects

    field_names : Optional[list[str]]
        Optional list of field names to analyze (if None, analyze all fields)

    Returns:
    --------
    dict
        Agreement report with metrics and conflicts
    """
    # Step 0: Kickout tasks with less than 2 coders
    initial_task_count = raw_annotations['task_id'].nunique()
    task_annotation_counts = raw_annotations.groupby('task_id')['ann_id'].nunique()
    tasks_with_multiple_anns = task_annotation_counts[task_annotation_counts > 1].index
    filtered_df = raw_annotations[raw_annotations['task_id'].isin(tasks_with_multiple_anns)]
    remaining_task_count = filtered_df['task_id'].nunique()
    print(f"Initial tasks: {initial_task_count}, Remaining tasks: {remaining_task_count}, "
          f"Removed tasks: {initial_task_count - remaining_task_count}")

    # Step 1: Create the assignment tracking DataFrame
    assignments_df = _create_assignment_tracking(assignments)

    # Step 2: Create annotations DataFrame without defaults
    annotations_df = create_annotations_dataframe(filtered_df)

    # Add image index information to identify indexed variables
    add_image_index_column(annotations_df)

    # Initialize results containers
    all_variable_agreements = {}
    all_conflicts = []

    # Create a mapping of base variable names to the original variables with indices
    base_to_indexed = {}
    if 'variable_base' in annotations_df.columns and 'image_idx' in annotations_df.columns:
        indexed_vars = annotations_df[annotations_df['image_idx'] > 0]
        for base_name in indexed_vars['variable_base'].unique():
            matching_vars = indexed_vars[indexed_vars['variable_base'] == base_name]['variable'].unique()
            if len(matching_vars) > 0:
                base_to_indexed[base_name] = list(matching_vars)

    # Process each variable individually
    for variable, variable_info in choices.items():
        if field_names and variable not in field_names:
            continue

        # Check if this is a base variable name that has indexed versions
        if variable in base_to_indexed:
            # We'll create a consolidated variable for all indexed versions
            consolidated_name = f"{variable}_$"

            # Get all annotations for the indexed versions of this variable
            all_indices_annotations = []
            for indexed_var in base_to_indexed[variable]:
                var_annotations = annotations_df[annotations_df['variable'] == indexed_var]
                if len(var_annotations) > 0:
                    all_indices_annotations.append(var_annotations)

            if all_indices_annotations:
                # Combine all indexed annotations
                variable_annotations = pd.concat(all_indices_annotations)

                # Apply defaults
                variable_with_defaults = apply_defaults_for_variable(
                    variable_annotations,
                    assignments_df,
                    consolidated_name,  # Use consolidated name
                    variable_info
                )

                # Create agreement matrix with task+index as the row identifier
                agreement_matrix = create_indexed_agreement_matrix(variable_with_defaults)

                # Calculate agreement
                variable_agreement = calculate_agreement(
                    agreement_matrix,
                    variable_info
                )

                # Store results under the consolidated name
                all_variable_agreements[consolidated_name] = variable_agreement

                # Identify conflicts
                variable_conflicts = identify_conflicts(
                    agreement_matrix,
                    consolidated_name,
                    variable_info,
                    variable_with_defaults
                )

                all_conflicts.extend(variable_conflicts)

        # Standard processing for regular variables
        else:
            variable_annotations = annotations_df[annotations_df['variable'] == variable]

            if len(variable_annotations) == 0:
                print(f"No data on {variable}")
                continue

            # Apply defaults
            variable_with_defaults = apply_defaults_for_variable(
                variable_annotations,
                assignments_df,
                variable,
                variable_info
            )

            # Create standard agreement matrix
            agreement_matrix = variable_with_defaults.pivot(
                index='task_id',
                columns='user_id',
                values='value'
            )

            # For single-choice, clear the agreement matrix
            if variable_info.choice == 'single':
                cleared_agreement_matrix = clear_agreement_matrix(agreement_matrix, variable_info)
                # Calculate agreement
                variable_agreement = calculate_agreement(
                    cleared_agreement_matrix,
                    variable_info
                )
            else:
                # For multiple-choice, use the original matrix
                variable_agreement = calculate_agreement(
                    agreement_matrix,
                    variable_info
                )

            # Store results
            all_variable_agreements[variable] = variable_agreement

            # Identify conflicts
            variable_conflicts = identify_conflicts(
                agreement_matrix,
                variable,
                variable_info,
                variable_with_defaults
            )

            all_conflicts.extend(variable_conflicts)

    # Generate the final report
    agreement_report = generate_agreement_report(
        all_variable_agreements,
        all_conflicts,
        choices,
        annotations_df,
        base_to_indexed
    )

    return agreement_report

def fix_users(df: DataFrame, usermap: dict) -> DataFrame:
    """
    Maps user IDs in the dataframe according to the provided mapping.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing a 'user_id' column

    usermap : dict
        Dictionary mapping user IDs to new identifiers

    Returns:
    --------
    DataFrame
        DataFrame with user_id column replaced according to the mapping
    """
    df['user_id'] = df['user_id'].map(usermap)
    return df


def generate_agreement_report(all_variable_agreements, all_conflicts, choices, annotations_df, base_to_indexed=None):
    """
    Generate final agreement report.

    Parameters:
    -----------
    all_variable_agreements : dict
        Dictionary of variable agreements

    all_conflicts : list
        List of conflicts

    choices : dict
        Variable metadata (mapping of variable_ids to ChoiceFieldModel objects)

    annotations_df : DataFrame
        Original annotations

    base_to_indexed : dict, optional
        Mapping of base variable names to their indexed versions

    Returns:
    --------
    dict with complete report
    """
    # Calculate overall statistics
    # Count unique tasks, coders, and variables
    unique_tasks = annotations_df['task_id'].nunique()
    unique_coders = annotations_df['user_id'].nunique()
    total_variables = len(choices)
    total_annotations = len(annotations_df)
    total_conflicts = len(all_conflicts)

    # Group variables by type
    single_choice_vars = []
    multiple_choice_vars = []

    for var_name, var_info in choices.items():
        if var_info.choice == 'single':
            single_choice_vars.append(var_name)
        elif var_info.choice == 'multiple':
            multiple_choice_vars.append(var_name)

    # Calculate overall agreement metrics
    single_choice_agreements = {}
    multiple_choice_agreements = {}

    # Calculate average agreements by choice type
    single_kappas = []
    single_gwets = []
    multiple_option_kappas = []
    multiple_option_gwets = []

    # Process single-choice variables
    for var_name, var_metrics in all_variable_agreements.items():
        var_info = None
        # Handle consolidated variables (with _$)
        if var_name.endswith('_$'):
            base_name = var_name[:-2]
            if base_name in choices:
                var_info = choices[base_name]
        else:
            if var_name in choices:
                var_info = choices[var_name]

        if var_info is None:
            continue

        if var_info.choice == 'single':
            # Extract kappa and gwet if available
            if 'kappa' in var_metrics:
                single_kappas.append(var_metrics['kappa'])
            if 'gwet' in var_metrics:
                single_gwets.append(var_metrics['gwet'])
            single_choice_agreements[var_name] = var_metrics
        elif var_info.choice == 'multiple':
            # For multiple-choice, we track option-level metrics
            if 'option_results' in var_metrics:
                for option, option_metrics in var_metrics['option_results'].items():
                    if 'kappa' in option_metrics:
                        multiple_option_kappas.append(option_metrics['kappa'])
                    if 'gwet' in option_metrics:
                        multiple_option_gwets.append(option_metrics['gwet'])
            multiple_choice_agreements[var_name] = var_metrics

    # Calculate averages
    avg_single_kappa = sum(single_kappas) / len(single_kappas) if single_kappas else 0
    avg_single_gwet = sum(single_gwets) / len(single_gwets) if single_gwets else 0
    avg_multiple_kappa = sum(multiple_option_kappas) / len(multiple_option_kappas) if multiple_option_kappas else 0
    avg_multiple_gwet = sum(multiple_option_gwets) / len(multiple_option_gwets) if multiple_option_gwets else 0

    # Calculate overall average across all variables
    all_kappas = single_kappas + multiple_option_kappas
    all_gwets = single_gwets + multiple_option_gwets
    overall_kappa = sum(all_kappas) / len(all_kappas) if all_kappas else 0
    overall_gwet = sum(all_gwets) / len(all_gwets) if all_gwets else 0

    # Process indexed variables
    indexed_vars_count = 0
    base_vars_count = 0
    if base_to_indexed:
        indexed_vars_count = sum(len(vars_list) for vars_list in base_to_indexed.values())
        base_vars_count = len(base_to_indexed)

    # Count conflicts by variable type
    single_choice_conflicts = [c for c in all_conflicts if
                               'conflict_type' not in c or c.get('conflict_type') != 'multiple_choice']
    multiple_choice_conflicts = [c for c in all_conflicts if
                                 'conflict_type' in c and c.get('conflict_type') == 'multiple_choice']

    # Calculate conflict rates
    single_conflict_rate = len(single_choice_conflicts) / (
                unique_tasks * len(single_choice_vars)) if unique_tasks > 0 and single_choice_vars else 0
    multiple_conflict_rate = len(multiple_choice_conflicts) / (
                unique_tasks * len(multiple_choice_vars)) if unique_tasks > 0 and multiple_choice_vars else 0
    overall_conflict_rate = total_conflicts / (
                unique_tasks * total_variables) if unique_tasks > 0 and total_variables > 0 else 0

    # Prepare the final report
    report = {
        "summary_stats": {
            "total_tasks": unique_tasks,
            "total_coders": unique_coders,
            "total_variables": total_variables,
            "total_annotations": total_annotations,
            "total_conflicts": total_conflicts,
            "single_choice_variables": len(single_choice_vars),
            "multiple_choice_variables": len(multiple_choice_vars),
            "indexed_variables_count": indexed_vars_count,
            "base_variables_count": base_vars_count,
            "conflict_rate": overall_conflict_rate,
        },
        "agreement_metrics": {
            "overall": {
                "kappa": float(overall_kappa),
                "gwet": float(overall_gwet),
            },
            "single_choice": {
                "average_kappa": float(avg_single_kappa),
                "average_gwet": float(avg_single_gwet),
                "variables": single_choice_agreements,
                "conflict_rate": float(single_conflict_rate),
                "conflict_count": len(single_choice_conflicts)
            },
            "multiple_choice": {
                "average_kappa": float(avg_multiple_kappa),
                "average_gwet": float(avg_multiple_gwet),
                "variables": multiple_choice_agreements,
                "conflict_rate": float(multiple_conflict_rate),
                "conflict_count": len(multiple_choice_conflicts)
            }
        },
        "conflicts": all_conflicts
    }

    # Make sure all numeric values are native Python types (not numpy types)
    # This ensures JSON serialization works correctly
    for key, value in report["summary_stats"].items():
        if hasattr(value, 'item'):
            report["summary_stats"][key] = value.item()

    return report

def create_indexed_agreement_matrix(variable_annotations):
    """
    Create an agreement matrix for variables with indices,
    using a composite key of task_id and image_idx.

    Parameters:
    -----------
    variable_annotations : DataFrame
        Annotations with image_idx column

    Returns:
    --------
    DataFrame
        Agreement matrix with composite row index
    """
    # Create a composite key
    if 'image_idx' in variable_annotations.columns:
        variable_annotations['composite_key'] = (
                variable_annotations['task_id'].astype(str) +
                '_' +
                variable_annotations['image_idx'].astype(str)
        )

        # Pivot table with composite key as index
        agreement_matrix = variable_annotations.pivot(
            index='composite_key',
            columns='user_id',
            values='value'
        )

        return agreement_matrix
    else:
        # Fall back to standard pivot if no image_idx
        return variable_annotations.pivot(
            index='task_id',
            columns='user_id',
            values='value'
        )

def agreement_matrix_assertions(agreement_matrix: DataFrame, variable_info: dict):
    # (index, data)
    for row in agreement_matrix.iterrows():
        # cell: (col, value)
        for idx, cell in enumerate(row[1]):
            if isinstance(cell, list) and len(cell) == 0:
                print(row[0], row[1].index[idx])


def add_image_index_column(df):
    """
    Extract indices from variable names and add columns for base variable name and index.
    For example: "nep_materiality_visual_0" -> base: "nep_materiality_visual", idx: 0

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing 'variable' or 'category' column

    Returns:
    --------
    dict
        Dictionary mapping base variable names to their indices
    """
    # Extract base names and indices
    base_names = []
    indices = []

    for var_name in df['variable']:
        var_str = str(var_name)
        # Look for _NUMBER at the end of the string
        match = re.search(r'_(\d+)(?:$|_)', var_str)

        if match:
            idx = int(match.group(0).strip("_"))
            # Remove the _NUMBER suffix to get base name
            if match.group(0)[0] == "_" and match.group(0)[-1] == "_":
                base = re.sub(r'_(\d+)(?:$|_)', '', var_str)
            else:
                base = re.sub(r'_(\d+)(?:$|_)', '', var_str)
            base_names.append(base)

            indices.append(idx)
        else:
            base_names.append(var_str)
            indices.append(0)  # 0 indicates no index

    # Add new columns
    df['variable_base'] = base_names
    df['image_idx'] = indices

    # Create mapping of base variables to their indices
    base_to_indices = {}
    for base, idx in zip(base_names, indices):
        if idx > 0:  # Only track variables with indices
            if base not in base_to_indices:
                base_to_indices[base] = []
            base_to_indices[base].append(idx)

    return base_to_indices


def clear_agreement_matrix(agreement_matrix: DataFrame, variable_info: ChoiceFieldModel) -> DataFrame:
    """
    Transforms agreement matrix for single-select variables by extracting values from lists.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    variable_info : ChoiceFieldModel
        Variable metadata including options

    Returns:
    --------
    DataFrame
        Cleared agreement matrix with indices instead of values
    """

    def single_val(c):
        if isinstance(c, list):
            return list(map(variable_info.option_index, c))
        elif isinstance(c, str) and c == NO_SINGLE_CHOICE:
            return [99]
        else:
            try:
                if np.isnan(c):
                    return np.NaN
            except:
                raise ValueError(f"maybe nan, for single. should be filled with default BEFORE: {c}")

    cleared_agreement_matrix = agreement_matrix.map(single_val)
    return cleared_agreement_matrix


def _create_assignment_tracking(assignments):
    """
    Create a DataFrame tracking all task assignments with metadata.

    Parameters:
    -----------
    assignments : list of dict or DataFrame
        Assignment data with task_id, coder_id, ann_id, ts, platform_id

    Returns:
    --------
    DataFrame with columns:
        - task_id, coder_id (index columns)
        - ann_id, timestamp, platform_id (metadata columns)
    """
    # Convert to DataFrame if needed
    if not isinstance(assignments, pd.DataFrame):
        assignments_df = pd.DataFrame(assignments)
    else:
        assignments_df = assignments

    # Keep the metadata columns
    metadata_cols = ['ann_id', 'ts', 'platform_id']
    keep_cols = ['task_id', 'user_id'] + [col for col in metadata_cols if col in assignments_df.columns]
    assignments_df = assignments_df[keep_cols]

    # Set task_id and user_id as index for efficient lookups
    assignments_df = assignments_df.set_index(['task_id', 'user_id'])

    return assignments_df


def create_annotations_dataframe(raw_annotations):
    """
    Create a DataFrame with all annotations.

    Parameters:
    -----------
    raw_annotations : list of dict or DataFrame
        Raw annotation data with task_id, ann_id, user_id, ts,
        platform_id, category, type, value

    Returns:
    --------
    DataFrame with annotation data
    """
    # Convert to DataFrame if needed
    if not isinstance(raw_annotations, pd.DataFrame):
        annotations_df = pd.DataFrame(raw_annotations)
    else:
        annotations_df = raw_annotations.copy()

    # Ensure consistent column names
    if 'category' in annotations_df.columns and 'variable' not in annotations_df.columns:
        annotations_df = annotations_df.rename(columns={'category': 'variable'})

    # Ensure user_id column exists
    if 'user_id' not in annotations_df.columns and 'coder_id' in annotations_df.columns:
        annotations_df = annotations_df.rename(columns={'coder_id': 'user_id'})

    return annotations_df


def apply_defaults_for_variable(variable_annotations: DataFrame,
                                assignments_df: DataFrame,
                                variable: str,
                                variable_info: ChoiceFieldModel):
    """
    Apply default values for a specific variable where annotations are missing.

    Parameters:
    -----------
    variable_annotations : DataFrame
        Annotations for this specific variable

    assignments_df : DataFrame
        Assignment tracking DataFrame

    variable : str
        ID of the variable

    variable_info : dict
        Metadata for this variable including type and default value

    Returns:
    --------
    DataFrame with defaults applied
    """
    # Reset index on assignments to get task_id and user_id as columns
    assignments = assignments_df.reset_index()

    # Create all possible task-coder combinations for this variable
    all_combinations = []
    for _, row in assignments.iterrows():
        combo = {
            'task_id': row['task_id'],
            'user_id': row['user_id'],
            'variable': variable,
            # Include metadata if available
            'ann_id': row.get('ann_id', None),
            'timestamp': row.get('ts', None),
            'platform_id': row.get('platform_id', None)
        }
        all_combinations.append(combo)

    all_combinations_df = pd.DataFrame(all_combinations)

    # Check if we need to handle image indices
    if 'image_idx' in variable_annotations.columns:
        # We need to handle each image index
        result_dfs = []

        for idx in variable_annotations['image_idx'].unique():
            # Filter annotations for this index
            idx_annotations = variable_annotations[variable_annotations['image_idx'] == idx]

            # Add image_idx to combinations
            idx_combinations = all_combinations_df.copy()
            idx_combinations['image_idx'] = idx

            # Merge
            merged = pd.merge(
                idx_combinations,
                idx_annotations,
                on=['task_id', 'user_id'],
                how='left',
                suffixes=('', '_existing')
            )

            # Keep the image_idx
            if 'image_idx_existing' in merged.columns and 'image_idx' in merged.columns:
                # Handle image_idx values carefully
                if 'image_idx_existing' in merged.columns:
                    # Create a mask for rows where image_idx_existing is null
                    mask = merged['image_idx_existing'].isna()
                    # Only fill those rows with values from image_idx
                    merged.loc[mask, 'image_idx'] = merged.loc[mask, 'image_idx']
                    # For all other rows, use image_idx_existing
                    merged.loc[~mask, 'image_idx'] = merged.loc[~mask, 'image_idx_existing']
                merged = merged.drop(columns=['image_idx_existing'])

            # Keep the variable_base if it exists
            if 'variable_base' in idx_annotations.columns:
                merged['variable_base'] = merged[
                    'variable_base_existing'] if 'variable_base_existing' in merged.columns else None
                if 'variable_base_existing' in merged.columns:
                    merged = merged.drop(columns=['variable_base_existing'])

            result_dfs.append(merged)

        if result_dfs:
            merged = pd.concat(result_dfs)
        else:
            merged = all_combinations_df
    else:
        # Standard merge without image indices
        merged = pd.merge(
            all_combinations_df,
            variable_annotations,
            on=['task_id', 'user_id'],
            how='left',
            suffixes=('', '_existing')
        )

    # Ensure the value column exists
    if 'value' not in merged.columns:
        logger.debug("Value column not found")
    else:
        if variable_info.choice == 'single':
            merged['value'] = merged['value'].apply(
                lambda x: x if isinstance(x, list) or not pd.isna(x) else [variable_info.safe_default]
            )
        else:  # multi_select
            merged['value'] = merged['value'].apply(
                lambda x: x if isinstance(x, list) or not pd.isna(x) else []
            )

    return merged


def calculate_agreement(agreement_matrix, variable_info):
    """
    Calculate agreement metrics for a variable.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    variable_info : dict
        Variable metadata including type and options

    Returns:
    --------
    dict with agreement metrics
    """
    # Handle different variable types
    if variable_info.choice == 'single':
        return _calculate_single_select_agreement2(agreement_matrix, variable_info)
    elif variable_info.choice == 'multiple':
        return _calculate_multi_select_agreement(agreement_matrix, variable_info)
    else:
        raise ValueError(f"Unknown variable type: {variable_info}")


def _calculate_single_select_agreement2(agreement_matrix,
                                        variable_info: ChoiceFieldModel,
                                        counts: bool = True,
                                        collect_conflicts_agreements: bool = True) -> dict:
    """Calculate agreement for single-select variables with improved metrics."""
    # Convert list values to single values
    agreement_matrix = agreement_matrix.map(lambda v: v[0] if isinstance(v, list) else np.NaN)

    # Calculate total valid rows (rows with at least one non-NaN value)
    valid_rows = agreement_matrix.dropna(how='all')
    total_valid_rows = len(valid_rows)

    # Calculate rows with agreement (all coders chose the same option)
    agreement_mask = valid_rows.apply(lambda row: row.dropna().nunique() == 1, axis=1)
    agreement_rows = valid_rows[agreement_mask]
    agreement_count = len(agreement_rows)

    # Calculate rows with disagreement
    disagreement_mask = valid_rows.apply(lambda row: row.dropna().nunique() > 1, axis=1)
    disagreement_rows = valid_rows[disagreement_mask]
    disagreement_count = len(disagreement_rows)

    # Calculate percent agreement (simpler metric than kappa)
    percent_agreement = agreement_count / total_valid_rows if total_valid_rows > 0 else 1.0

    # Calculate Fleiss' kappa
    try:
        cac = CAC(agreement_matrix)
        fk = cac.fleiss()
        kappa = fk["est"]["coefficient_value"]
        gwet = cac.gwet()["est"]["coefficient_value"]
    except Exception as e:
        logger.error(f"Kappa calculation error for {variable_info.name}: {str(e)}")
        kappa = 1 if agreement_count == total_valid_rows else 0
        gwet = 0

    # Option counts - count exactly how many times each option was selected
    option_counts = {}
    flat_values = agreement_matrix.values.flatten()
    series_counts = pd.Series(flat_values).value_counts().to_dict()

    for option in variable_info.options:
        option_idx = variable_info.option_index(option)
        # Convert NumPy types to Python native types
        count = series_counts.get(option_idx, 0)
        option_counts[option] = int(count) if hasattr(count, 'item') else count

    # Get conflicts if needed
    conflicts = None
    if collect_conflicts_agreements:
        conflicts = disagreement_rows.index.tolist()

    # Per-option detailed analysis
    option_results = {}
    for option in variable_info.options:
        option_idx = variable_info.option_index(option)

        # Rows where this option appears at least once
        option_mask = (agreement_matrix == option_idx).any(axis=1)
        option_rows = agreement_matrix[option_mask]
        option_row_count = len(option_rows)

        # Rows with complete agreement on this option
        agreement_on_option_mask = option_rows.apply(
            lambda row: row.dropna().nunique() == 1 and row.dropna().iloc[0] == option_idx,
            axis=1
        )
        agreement_on_option_count = agreement_on_option_mask.sum()

        # Rows with disagreement involving this option
        disagreement_on_option_count = option_row_count - agreement_on_option_count
        error = None
        # Option-specific kappa
        try:
            if len(option_rows) > 0:
                cac = CAC(option_rows)
                option_kappa = cac.fleiss()["est"]["coefficient_value"]
                option_gwet = cac.gwet()["est"]["coefficient_value"]
            else:
                option_kappa = 1.0
                option_gwet = 1
        except Exception as e:
            option_kappa = 1.0 if disagreement_on_option_count == 0 else 0.0
            option_gwet = 1
            error = str(e)

        # Convert any NumPy types to native Python types
        option_results[option] = {
            "error": error,
            "kappa": float(option_kappa),
            "gwet": float(option_gwet),
            "number": int(option_row_count) if hasattr(option_row_count, 'item') else option_row_count,
            "agreement_count": int(agreement_on_option_count) if hasattr(agreement_on_option_count,
                                                                         'item') else agreement_on_option_count,
            "disagreement_count": int(disagreement_on_option_count) if hasattr(disagreement_on_option_count,
                                                                               'item') else disagreement_on_option_count,
            "total_selections": option_counts[option]
        }

    # Final conversion of all values to JSON-serializable types
    return {
        'kappa': float(kappa) if hasattr(kappa, 'item') else kappa,
        "gwet": float(gwet) if hasattr(gwet, 'item') else gwet,
        'percent_agreement': float(percent_agreement) if hasattr(percent_agreement, 'item') else percent_agreement,
        'total_rows': int(total_valid_rows) if hasattr(total_valid_rows, 'item') else total_valid_rows,
        'agreement_count': int(agreement_count) if hasattr(agreement_count, 'item') else agreement_count,
        'disagreement_count': int(disagreement_count) if hasattr(disagreement_count, 'item') else disagreement_count,
        'counts': option_counts,
        'option_results': option_results,
        'conflicts': [str(c) for c in conflicts] if conflicts is not None else None
    }


def _calculate_multi_select_agreement(agreement_matrix, variable_info: ChoiceFieldModel,
                                      counts: bool = True, collect_conflicts_agreements: bool = True):
    """
    Calculate agreement for multi-select variables by creating
    binary matrices for each option.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    variable_info : ChoiceFieldModel
        Variable metadata including type and options

    counts : bool
        Whether to collect counts of options

    collect_conflicts_agreements : bool
        Whether to collect conflicts

    Returns:
    --------
    dict with agreement metrics per option
    """
    options = variable_info.options

    # Track agreement metrics for each option
    option_results = {}
    all_conflicts = []

    # Get overall counts if needed
    option_counts = {}
    if counts:
        # Count how many times each option was selected
        for option in options:
            count = 0
            for _, row in agreement_matrix.iterrows():
                for value in row.dropna():
                    if isinstance(value, list) and option in value:
                        count += 1
            option_counts[option] = count

    # Process each option separately as a binary choice
    for option in options:
        # Create binary matrix for this option (1 if present, 0 if absent)
        binary_matrix = agreement_matrix.applymap(
            lambda x: 1 if isinstance(x, list) and option in x else 0
        )

        # Calculate metrics for valid rows (rows with at least one non-NaN value)
        valid_rows = binary_matrix.dropna(how='all')
        total_valid_rows = len(valid_rows)

        # Calculate rows with agreement (all coders chose the same)
        agreement_mask = valid_rows.apply(lambda row: row.dropna().nunique() == 1, axis=1)
        agreement_rows = valid_rows[agreement_mask]
        agreement_count = len(agreement_rows)

        # Calculate rows with disagreement
        disagreement_mask = valid_rows.apply(lambda row: row.dropna().nunique() > 1, axis=1)
        disagreement_rows = valid_rows[disagreement_mask]
        disagreement_count = len(disagreement_rows)

        # Calculate percent agreement
        percent_agreement = agreement_count / total_valid_rows if total_valid_rows > 0 else 1.0

        # Calculate kappa and gwet for this option
        kappa = 1.0
        gwet = 1.0
        error = None

        try:
            if len(valid_rows) > 0:
                cac = CAC(binary_matrix)
                kappa = cac.fleiss()["est"]["coefficient_value"]
                gwet = cac.gwet()["est"]["coefficient_value"]
        except Exception as e:
            logger.error(f"Agreement calculation error for option {option}: {str(e)}")
            error = str(e)

        # Get conflicts for this option if needed
        option_conflicts = None
        if collect_conflicts_agreements:
            option_conflicts = disagreement_rows.index.tolist()

            # Add conflict details to all_conflicts
            for task_id in option_conflicts:
                row_data = binary_matrix.loc[task_id].dropna()
                conflict_info = {
                    'task_id': task_id,
                    'option': option,
                    'agreement_score': 0,  # 0 because it's a disagreement
                    'annotations': [
                        {'user_id': coder, 'value': 'present' if value == 1 else 'absent'}
                        for coder, value in row_data.items()
                    ]
                }
                all_conflicts.append(conflict_info)

        # Count rows where this option was selected at least once
        rows_with_option = binary_matrix[binary_matrix.eq(1).any(axis=1)]
        option_presence_count = len(rows_with_option)

        # Count rows where all coders agreed this option was present
        all_present_mask = binary_matrix.apply(lambda row: all(val == 1 for val in row.dropna()), axis=1)
        all_present_count = all_present_mask.sum()

        # Count rows where all coders agreed this option was absent
        all_absent_mask = binary_matrix.apply(lambda row: all(val == 0 for val in row.dropna()), axis=1)
        all_absent_count = all_absent_mask.sum()

        # Store metrics for this option
        option_results[option] = {
            'error': error,
            'kappa': float(kappa),
            'gwet': float(gwet),
            'percent_agreement': float(percent_agreement),
            'total_rows': int(total_valid_rows),
            'agreement_count': int(agreement_count),
            'disagreement_count': int(disagreement_count),
            'option_presence_count': int(option_presence_count),
            'all_present_count': int(all_present_count),
            'all_absent_count': int(all_absent_count),
            'conflicts': [str(c) for c in option_conflicts] if option_conflicts is not None else None
        }

    # Final result - no overall kappa, just per-option statistics
    return {
        'counts': option_counts,
        'option_results': option_results,
        'conflicts': all_conflicts
    }