import pandas as pd
import re
import numpy as np
from irrCAC.raw import CAC
from pandas import DataFrame


def analyze_coder_agreement(raw_annotations, assignments, variables) -> dict:
    """
    End-to-end function to analyze coder agreement across annotations.

    Parameters:
    -----------
    raw_annotations : list of dict or DataFrame
        List of annotation objects with keys:
        task_id, ann_id, coder_id, ts, platform_id, category (variable name), type, value

    assignments : list of dict or DataFrame
        List of task-coder assignments with metadata

    variables : dict
        Dictionary mapping variable_ids to their metadata:
        {
            "variable_id": {
                "type": "single_select" or "multi_select",
                "options": [list of allowed values],
                "default": default_value
            }
        }

    Returns:
    --------
    dict
        Agreement report with metrics and conflicts
    """
    # Step 1: Create the assignment tracking DataFrame
    assignments_df = _create_assignment_tracking(assignments)

    # Step 2: Create annotations DataFrame without defaults
    annotations_df = create_annotations_dataframe(raw_annotations)

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
    for variable, variable_info in variables.items():
        # Skip text variables
        if variable_info["type"] == "text":
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
            agreement_matrix_assertions(agreement_matrix, variable_info)
            cleared_agreement_matrix = clear_agreement_matrix(agreement_matrix, variable_info)
            # Calculate agreement
            variable_agreement = calculate_agreement(
                cleared_agreement_matrix,
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
        variables,
        annotations_df,
        base_to_indexed
    )

    return agreement_report


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
    # Determine which column to use
    # if 'variable' not in df.columns and 'category' in df.columns:
    #     df['variable'] = df['category']
    #
    # if 'variable' not in df.columns:
    #     print("Warning: No variable column found")
    #     return {}

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

    # Print summary
    if base_to_indices:
        print(f"Found {len(base_to_indices)} variables with indices:")
        for base, idxs in base_to_indices.items():
            print(f"  {base}: indices {sorted(idxs)}")

    return base_to_indices


def clear_agreement_matrix(agreement_matrix: DataFrame, variable_info):
    def single_val(c):
        if isinstance(c, list):
            if len(c) == 0:
                if variable_info["type"] == "multiple":
                    return []
                else:
                    print(f"something strange {variable_info}: {c}. Filling in default")
                    return variable_info["options"].index(variable_info["default"])
            elif len(c) == 1:
                return variable_info["options"].index(c[0])
            else:
                assert variable_info["type"] == "multiple"
                return list(map(variable_info["options"].index, c))

        else:
            return c

    cleared_agreement_matrix = agreement_matrix.map(single_val)
    return cleared_agreement_matrix


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


def apply_defaults_for_variable(variable_annotations, assignments_df, variable, variable_info):
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
        # Create default values based on variable type
        if variable_info['type'] == "text":
            merged['value'] = ""
        elif variable_info['type'] == 'single':
            merged['value'] = variable_info.get('default', None)
        else:  # multi_select
            merged['value'] = [[] for _ in range(len(merged))]
    else:
        # Apply defaults to missing values
        if variable_info['type'] == "text":
            merged['value'] = merged['value'].fillna(value="")
        elif variable_info['type'] == 'single_select':
            merged['value'] = merged['value'].fillna(value=variable_info.get('default', None))
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

    # Standardize variable type strings
    if variable_info["type"] in ['single']:
        return _calculate_single_select_agreement(agreement_matrix, variable_info['options'])
    elif variable_info["type"] in ['multiple']:
        return _calculate_multi_select_agreement(agreement_matrix, variable_info['options'])
    else:
        raise ValueError(f"Unknown variable type: {variable_info}")


def _calculate_single_select_agreement(agreement_matrix, options):
    """Calculate agreement for single-select variables."""
    # Get coder pairs

    cac = CAC(agreement_matrix)

    try:
        fk = cac.fleiss()
        res = fk["est"]["coefficient_value"]
    except:
        res = 1
    return {
        # 'overall_agreement': overall_agreement,
        'kappa': res,
        # 'sample_size': total_comparisons
    }


def _calculate_multi_select_agreement(agreement_matrix, options):
    """
    Calculate agreement for multi-select variables by creating
    binary matrices for each option.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    options : list
        List of allowed options for this variable

    Returns:
    --------
    dict with agreement metrics
    """
    # Track agreement metrics for each option
    option_metrics = {}
    overall_kappas = []

    # Process each option separately
    for option in options:
        # Create binary matrix for this option
        binary_matrix = agreement_matrix.applymap(
            lambda x: 1 if isinstance(x, list) and option in x else 0
        )

        # Calculate agreement for this option (treat as binary single-select)
        option_agreement = _calculate_binary_agreement(binary_matrix)
        option_metrics[option] = option_agreement
        overall_kappas.append(option_agreement.get('kappa', 0))

    # Calculate overall metrics
    avg_kappa = sum(overall_kappas) / len(overall_kappas) if overall_kappas else 0

    return {
        'overall_agreement': avg_kappa,
        'kappa_by_option': option_metrics,
        'average_kappa': avg_kappa
    }


def _calculate_binary_agreement(binary_matrix):
    """
    Calculate agreement for a binary matrix (presence/absence of an option).

    Parameters:
    -----------
    binary_matrix : DataFrame
        Matrix with tasks as rows, coders as columns, and binary values

    Returns:
    --------
    dict with agreement metrics
    """
    # Get coder pairs
    coders = binary_matrix.columns
    n_coders = len(coders)

    # Track agreements and kappa values
    total_agreements = 0
    total_comparisons = 0
    kappa_values = []

    for i in range(n_coders):
        for j in range(i + 1, n_coders):
            coder1 = coders[i]
            coder2 = coders[j]

            # Get data for this pair
            pair_data = binary_matrix[[coder1, coder2]].dropna()

            if len(pair_data) == 0:
                continue

            # Count agreements
            agreements = (pair_data[coder1] == pair_data[coder2]).sum()
            total_agreements += agreements
            total_comparisons += len(pair_data)

            # Calculate Cohen's kappa for this binary choice
            observed_agreement = agreements / len(pair_data)

            # Calculate expected agreement (simpler for binary case)
            prob_coder1_yes = pair_data[coder1].mean()
            prob_coder1_no = 1 - prob_coder1_yes
            prob_coder2_yes = pair_data[coder2].mean()
            prob_coder2_no = 1 - prob_coder2_yes

            expected_agreement = (prob_coder1_yes * prob_coder2_yes) + (prob_coder1_no * prob_coder2_no)

            # Calculate kappa
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                kappa_values.append(kappa)

    # Overall metrics
    overall_agreement = total_agreements / total_comparisons if total_comparisons > 0 else 0
    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0

    return {
        'agreement_rate': overall_agreement,
        'kappa': avg_kappa,
        'sample_size': total_comparisons
    }


def identify_conflicts(agreement_matrix, variable, variable_info, variable_annotations):
    """
    Identify conflicts for this variable.

    Parameters:
    -----------
    agreement_matrix : DataFrame
        Matrix with tasks as rows and coders as columns

    variable : str
        ID of the variable

    variable_info : dict
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
    has_composite_key = isinstance(agreement_matrix.index[0], str) and '_' in agreement_matrix.index[0] if len(
        agreement_matrix.index) > 0 else False

    variable_type = variable_info.get('type', '')
    if variable_type in ['single_select', 'single']:
        # For single-select, check exact matches
        for task_id, row in agreement_matrix.iterrows():
            values = row.dropna().tolist()
            coders = row.dropna().index.tolist()

            if len(values) < 2:
                continue  # Need at least 2 annotations

            # Check if all values are equal
            if len(set(str(val) for val in values)) == 1:  # Convert to strings for comparison
                continue  # All values are the same, no conflict

            # Calculate agreement score (0 for conflict, 1 for agreement)
            agreement = 0.0  # Since we know there's a disagreement at this point

            # Extract actual task_id and image_idx if using composite keys
            actual_task_id = task_id
            image_idx = None

            if has_composite_key:
                parts = task_id.split('_')
                if len(parts) >= 2:
                    actual_task_id = parts[0]
                    try:
                        image_idx = int(parts[1])
                    except (ValueError, IndexError):
                        pass

            # Find platform_id for this task
            platform_id = platform_id_map.get(actual_task_id)

            # Record conflict
            conflict = {
                'task_id': actual_task_id,
                'platform_id': platform_id,
                'variable': variable,
                'agreement_score': agreement,
                'annotations': [
                    {'user_id': coders[i], 'value': values[i]}
                    for i in range(len(coders))
                ]
            }

            # Add image_idx if available
            if image_idx is not None:
                conflict['image_idx'] = image_idx

            conflicts.append(conflict)
    else:
        # For multi-select, check option-by-option
        options = variable_info.get('options', [])

        for task_id, row in agreement_matrix.iterrows():
            values = row.dropna().tolist()
            coders = row.dropna().index.tolist()

            if len(values) < 2:
                continue  # Need at least 2 annotations

            # Check if all values are identical (including empty lists)
            serialized_values = [str(sorted(v) if isinstance(v, list) else v) for v in values]
            if len(set(serialized_values)) == 1:
                continue  # All values are the same, no conflict

            # Calculate option-by-option agreement
            option_agreements = []
            for option in options:
                # Check if coders agree on this option
                option_values = [1 if isinstance(v, list) and option in v else 0 for v in values]
                option_agreement = 1.0 if len(set(option_values)) == 1 else 0.0
                option_agreements.append(option_agreement)

            # Average agreement across options
            avg_agreement = sum(option_agreements) / len(option_agreements) if option_agreements else 0

            # Extract actual task_id and image_idx if using composite keys
            actual_task_id = task_id
            image_idx = None

            if has_composite_key:
                parts = task_id.split('_')
                if len(parts) >= 2:
                    actual_task_id = parts[0]
                    try:
                        image_idx = int(parts[1])
                    except (ValueError, IndexError):
                        pass

            # Record conflict
            conflict = {
                'task_id': actual_task_id,
                'platform_id': platform_id_map.get(actual_task_id, None),
                'variable': variable,
                'agreement_score': avg_agreement,
                'annotations': [
                    {'user_id': coders[i], 'value': values[i]}
                    for i in range(len(coders))
                ]
            }

            # Add image_idx if available
            if image_idx is not None:
                conflict['image_idx'] = image_idx

            conflicts.append(conflict)

    return conflicts


def generate_agreement_report(all_variable_agreements, all_conflicts, variables, annotations_df, base_to_indexed=None):
    """
    Generate final agreement report.

    Parameters:
    -----------
    all_variable_agreements : dict
        Dictionary of variable agreements

    all_conflicts : list
        List of conflicts

    variables : dict
        Variable metadata

    annotations_df : DataFrame
        Original annotations

    base_to_indexed : dict, optional
        Mapping of base variable names to their indexed versions

    Returns:
    --------
    dict with complete report
    """
    # Calculate overall metrics
    all_scores = [metrics.get('overall_agreement', 0)
                  for metrics in all_variable_agreements.values()]

    overall_agreement = sum(all_scores) / len(all_scores) if all_scores else 0

    # Count unique tasks and coders
    unique_tasks = annotations_df['task_id'].nunique()
    unique_coders = annotations_df['user_id'].nunique()

    # Count consolidated variables (those ending with _$)
    consolidated_vars = sum(1 for var in all_variable_agreements if var.endswith('_$'))

    # Get counts of indexed variables
    indexed_vars_count = 0
    base_vars_count = 0

    if base_to_indexed:
        indexed_vars_count = sum(len(vars_list) for vars_list in base_to_indexed.values())
        base_vars_count = len(base_to_indexed)

    # Build report
    report = {
        "agreement_metrics": {
            "overall": overall_agreement,
            "by_variable": all_variable_agreements
        },
        "conflicts": all_conflicts,
        "summary_stats": {
            "total_tasks": unique_tasks,
            "total_variables": len(variables),
            "total_coders": unique_coders,
            "total_annotations": len(annotations_df),
            "consolidated_variables": consolidated_vars,
            "indexed_variables_count": indexed_vars_count,
            "base_variables_count": base_vars_count,
            "conflict_rate": len(all_conflicts) / (unique_tasks * len(variables)) if unique_tasks > 0 and len(
                variables) > 0 else 0
        }
    }

    return report


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
