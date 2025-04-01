import pandas as pd
from pandas import DataFrame


def analyze_coder_agreement(raw_annotations, assignments, variables) -> dict:
    """
    End-to-end function to analyze coder agreement across annotations.

    Parameters:
    -----------
    raw_annotations : list of dict
        List of annotation objects with keys:
        task_id, ann_id, coder_id, ts, platform_id, category (variable name), type, value

    assignments : list of dict
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

    # Initialize results containers
    all_variable_agreements = {}
    all_conflicts = []

    # Step 3: Process each variable individually
    for variable, variable_info in variables.items():
        # Filter annotations for just this variable
        if variable_info["type"] == "text":
            continue
        variable_annotations = annotations_df[annotations_df['variable'] == variable]

        # Apply defaults for just this variable
        variable_with_defaults = apply_defaults_for_variable(
            variable_annotations,
            assignments_df,
            variable,
            variable_info
        )

        # Create agreement matrix (task rows, coder columns)
        agreement_matrix = _create_agreement_matrix(
            variable_with_defaults
        )

        # Calculate agreement for this variable

        variable_agreement = calculate_agreement(
            agreement_matrix,
            variable_info
        )

        # Store results
        # all_variable_agreements[variable] = variable_agreement

        # Identify conflicts for this variable
        variable_conflicts = identify_conflicts(
            agreement_matrix,
            variable,
            variable_info,
            variable_with_defaults  # Pass the annotations with metadata
        )

        # Add to overall conflicts
        all_conflicts.extend(variable_conflicts)

    # Generate the final report
    agreement_report = generate_agreement_report(
        all_variable_agreements,
        all_conflicts,
        variables,
        annotations_df
    )

    return agreement_report


def _create_assignment_tracking(assignments):
    """
    Create a DataFrame tracking all task assignments with metadata.

    Parameters:
    -----------
    assignments : list of dict
        Assignment data with task_id, coder_id, ann_id, ts, platform_id

    Returns:
    --------
    DataFrame with columns:
        - task_id, coder_id (index columns)
        - ann_id, timestamp, platform_id (metadata columns)
    """
    assignments_df = pd.DataFrame(assignments)

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
    raw_annotations : list of dict
        Raw annotation data with task_id, ann_id, user_id, ts,
        platform_id, category, type, value

    Returns:
    --------
    DataFrame with annotation data
    """
    # Convert to DataFrame
    annotations_df = pd.DataFrame(raw_annotations)

    # Ensure consistent column names
    if 'category' in annotations_df.columns and 'variable' not in annotations_df.columns:
        annotations_df = annotations_df.rename(columns={'category': 'variable'})

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
        all_combinations.append({
            'task_id': row['task_id'],
            'user_id': row['user_id'],
            'variable': variable,
            # Include metadata if available
            'ann_id': row.get('ann_id', None),
            'timestamp': row.get('ts', None),
            'platform_id': row.get('platform_id', None)
        })

    all_combinations_df = pd.DataFrame(all_combinations)

    # Merge with existing annotations, keeping all combinations
    merged = pd.merge(
        all_combinations_df,
        variable_annotations,
        on=['task_id', 'user_id', 'variable'],
        how='left',
        suffixes=('', '_existing')
    )

    if variable_info['type'] == "text":
        merged['value'].fillna("")
        return merged
    # Apply defaults where values are missing
    if variable_info['type'] == 'single_select':
        merged['value'] = merged['value'].fillna(variable_info['default'])
    else:  # multi_select
        # For multi-select, empty list is the default
        # Fixed version: Use pandas.isna() on scalar values in apply function
        merged['value'] = merged['value'].apply(
            lambda x: x if isinstance(x, list) or not pd.isna(x) else []
        )

    return merged


def _create_agreement_matrix(variable_annotations):
    """
    Create a matrix with tasks as rows and coders as columns.

    Parameters:
    -----------
    variable_annotations : DataFrame
        Annotations for this variable with defaults applied

    Returns:
    --------
    DataFrame with tasks as rows and coders as columns
    """
    # Pivot the data
    agreement_matrix = variable_annotations.pivot(
        index='task_id',
        columns='user_id',
        values='value'
    )

    return agreement_matrix


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
    if variable_info['type'] == 'single_select':
        return _calculate_single_select_agreement(agreement_matrix, variable_info['options'])
    else:  # multi_select
        return _calculate_multi_select_agreement(agreement_matrix, variable_info['options'])


def _calculate_single_select_agreement(agreement_matrix, options):
    """
    Calculate agreement for single-select variables.

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
    # Get coder pairs
    coders = agreement_matrix.columns
    n_coders = len(coders)

    # Track agreements
    total_agreements = 0
    total_comparisons = 0

    # Track kappa values
    kappa_values = []

    for i in range(n_coders):
        for j in range(i + 1, n_coders):
            coder1 = coders[i]
            coder2 = coders[j]

            # Get annotations for this pair
            pair_data = agreement_matrix[[coder1, coder2]].dropna()

            if len(pair_data) == 0:
                continue

            # Count agreements
            agreements = (pair_data[coder1] == pair_data[coder2]).sum()
            total_agreements += agreements
            total_comparisons += len(pair_data)

            # Calculate Cohen's kappa
            observed_agreement = agreements / len(pair_data)

            # Calculate expected agreement
            expected_agreement = 0
            for option in options:
                prob_coder1 = (pair_data[coder1] == option).mean()
                prob_coder2 = (pair_data[coder2] == option).mean()
                expected_agreement += prob_coder1 * prob_coder2

            # Calculate kappa
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                kappa_values.append(kappa)

    # Overall metrics
    overall_agreement = total_agreements / total_comparisons if total_comparisons > 0 else 0
    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0

    return {
        'overall_agreement': overall_agreement,
        'kappa': avg_kappa,
        'sample_size': total_comparisons
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
        binary_matrix = agreement_matrix.map(
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

    threshold : float
        Agreement threshold

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

    # For debugging
    #print(f"Platform ID map: {platform_id_map}")

    if variable_info['type'] == 'single_select':
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

            # Find platform_id for this task
            platform_id = platform_id_map.get(task_id)

            # Record conflict
            conflict = {
                'task_id': task_id,
                'platform_id': platform_id,
                'variable': variable,
                'agreement_score': agreement,
                'annotations': [
                    {'user_id': coders[i], 'value': values[i]}
                    for i in range(len(coders))
                ]
            }
            conflicts.append(conflict)
    else:
        # For multi-select, check option-by-option
        options = variable_info['options']

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

            # If below threshold, record conflict
            conflict = {
                'task_id': task_id,
                'platform_id': platform_id_map.get(task_id, None),
                'variable': variable,
                'agreement_score': avg_agreement,
                'annotations': [
                    {'user_id': coders[i], 'value': values[i]}
                    for i in range(len(coders))
                ]
            }
            conflicts.append(conflict)

    return conflicts


def generate_agreement_report(all_variable_agreements, all_conflicts, variables, annotations_df):
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
            "conflict_rate": len(all_conflicts) / (unique_tasks * len(variables)) if unique_tasks > 0 else 0
        }
    }

    return report

def fix_users(df: DataFrame, usermap: dict[int,str]) -> DataFrame:
    df['user_id'] = df['user_id'].map(usermap)
    return df