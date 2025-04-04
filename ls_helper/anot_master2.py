import re

import numpy as np
import pandas as pd


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


def analyze_coder_agreement(raw_annotations, assignments, variables):
    """
    Calculate agreement between coders, consolidating variables with indices.

    Parameters:
    -----------
    raw_annotations : list or DataFrame
        Raw annotation data
    assignments : list or DataFrame
        Assignment data
    variables : dict
        Variable metadata

    Returns:
    --------
    dict
        Agreement report
    """
    # Convert inputs to DataFrames if needed
    annotations_df = raw_annotations.copy()

    # if not isinstance(assignments, pd.DataFrame):
    #     assignments_df = pd.DataFrame(assignments)
    # else:
    #     assignments_df = assignments.copy()

    # Ensure required columns
    assert annotations_df.columns.to_list() == ['task_id', 'ann_id', 'user_id', 'platform_id', 'ts', 'type', 'category','value']
    if 'category' in annotations_df.columns and 'variable' not in annotations_df.columns:
        annotations_df = annotations_df.rename(columns={'category': 'variable'})

    # if 'coder_id' in annotations_df.columns and 'user_id' not in annotations_df.columns:
    #     annotations_df = annotations_df.rename(columns={'coder_id': 'user_id'})

    # Add index information to variables
    add_image_index_column(annotations_df)
    # print(base_to_indices)
    # Create results containers
    all_agreements = {}
    all_conflicts = []

    # Group variables by base name for consolidation
    indexed_variables = {}
    pass
    # Get all variables with indices > 0
    indexed_vars = annotations_df[annotations_df['image_idx'] > 0]
    for base_name in indexed_vars['variable_base'].unique():
       indexed_variables[base_name] = indexed_vars[indexed_vars['variable_base'] == base_name]['variable'].unique()

    # Process consolidated variables first
    for base_var, indexed_vars_list in indexed_variables.items():
        # Check if variable metadata exists (as base_name or base_name_$)
        var_key = None
        if base_var + "_$" in variables:
            var_key = base_var + "_$"
        elif base_var in variables:
            var_key = base_var

        if var_key is None:
            print(f"Skipping {base_var} - no variable metadata found")
            continue

        var_info = variables[var_key]

        # Skip text variables
        if var_info.get('type') in ['text']:
            continue

        print(
            f"Processing consolidated variable {base_var} from {len(indexed_vars_list)} indexed variables: {', '.join(indexed_vars_list)}")

        # Get all annotations for the indexed versions
        indexed_annotations = annotations_df[annotations_df['variable'].isin(indexed_vars_list)]

        if len(indexed_annotations) == 0:
            continue

        # Create a composite key for task_id + image_idx
        indexed_annotations['composite_key'] = (
                indexed_annotations['task_id'].astype(str) + '_' +
                indexed_annotations['image_idx'].astype(str)
        )

        # Create agreement matrix
        matrix = indexed_annotations.pivot(
            index='composite_key',
            columns='user_id',
            values='value'
        )

        # Calculate agreement
        agreement = calculate_agreement(matrix, var_info)

        # Store as consolidated variable with _$ suffix
        all_agreements[base_var + "_$"] = agreement

        print(f"  Agreement for {base_var}_$: {agreement}")

        # Find conflicts
        conflicts = find_conflicts(matrix, base_var + "_$", var_info, indexed_annotations)
        all_conflicts.extend(conflicts)

    # Process regular variables (not consolidated)
    for var_name, var_info in variables.items():
        # Skip if this is a consolidated variable or already processed
        if var_name.endswith('_$'):
            continue

        # Skip if this is a base variable that has indexed versions
        if var_name in indexed_variables:
            continue

        # Skip text variables
        if var_info.get('type') in ['text']:
            continue

        # Get annotations for this variable
        var_annotations = annotations_df[annotations_df['variable'] == var_name]

        if len(var_annotations) == 0:
            continue

        print(f"Processing regular variable {var_name}")

        # Create agreement matrix
        matrix = var_annotations.pivot(
            index='task_id',
            columns='user_id',
            values='value'
        )

        # Calculate agreement
        agreement = calculate_agreement(matrix, var_info)

        # Store results
        all_agreements[var_name] = agreement

        print(f"Agreement for {var_name}: {agreement}")

        # Find conflicts
        conflicts = find_conflicts(matrix, var_name, var_info, var_annotations)
        all_conflicts.extend(conflicts)

    # Generate final report
    consolidated_vars = {k: v for k, v in all_agreements.items() if k.endswith('_$')}
    regular_vars = {k: v for k, v in all_agreements.items() if not k.endswith('_$')}

    report = {
        "agreement_metrics": {
            "by_variable": all_agreements,
            "consolidated_variables": consolidated_vars,
            "regular_variables": regular_vars
        },
        "conflicts": all_conflicts,
        "summary": {
            "total_variables": len(all_agreements),
            "consolidated_variables": len(consolidated_vars),
            "regular_variables": len(regular_vars),
            "indexed_variables": sum(len(vars_list) for vars_list in indexed_variables.values())
        }
    }

    print("\nFinal report:")
    print(f"Processed {len(all_agreements)} variables total")
    print(f"Consolidated variables ({len(consolidated_vars)}): {list(consolidated_vars.keys())}")

    return report


def calculate_agreement(agreement_matrix, variable_info):
    """Calculate agreement metrics based on variable type."""
    var_type = variable_info.get('type', '')

    # Handle different naming conventions for variable types
    if var_type in ['single', 'single_select']:
        return calculate_single_select_agreement(agreement_matrix, variable_info.get('options', []))
    elif var_type in ['multi', 'multi_select']:
        return calculate_multi_select_agreement(agreement_matrix, variable_info.get('options', []))
    else:
        print(f"Unknown variable type: {var_type}, treating as single_select")
        return calculate_single_select_agreement(agreement_matrix, variable_info.get('options', []))


def calculate_single_select_agreement(agreement_matrix, options):
    """Calculate agreement for single-select variables."""
    # Get coder pairs
    coders = agreement_matrix.columns
    n_coders = len(coders)

    # Track metrics
    total_agreements = 0
    total_comparisons = 0
    kappa_values = []

    # Compare each pair of coders
    for i in range(n_coders):
        for j in range(i + 1, n_coders):
            coder1 = coders[i]
            coder2 = coders[j]

            # Get rows where both coders provided answers
            pair_data = agreement_matrix[[coder1, coder2]].dropna()

            if len(pair_data) == 0:
                continue

            # Count exact agreements
            agreements = (pair_data[coder1] == pair_data[coder2]).sum()
            total_agreements += agreements
            total_comparisons += len(pair_data)

            # Calculate Cohen's kappa
            observed_agreement = agreements / len(pair_data)

            # Calculate expected agreement by chance
            expected_agreement = 0
            for option in options:
                prob_coder1 = (pair_data[coder1] == option).mean()
                prob_coder2 = (pair_data[coder2] == option).mean()
                expected_agreement += prob_coder1 * prob_coder2

            # Compute kappa if possible
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                kappa_values.append(kappa)

    # Calculate overall metrics
    overall_agreement = total_agreements / total_comparisons if total_comparisons > 0 else 0
    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0

    def single_val(c):
        if isinstance(c, list):
            if len(c) == 0:
                return np.nan
            elif len(c) == 1:
                if c[0] not in options:
                    return c[0]
                return options.index(c[0])
        else:
            return c

    """"
    agreement_matrix.map(
        lambda c: options.index(c[0]) if isinstance(c, list) and len(c) > 0 and c[0] in options else np.nan)
    """
    agreement_matrix2 = agreement_matrix.map(single_val)

    from irrCAC.raw import CAC
    cac = CAC(agreement_matrix2)

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


def calculate_multi_select_agreement(agreement_matrix, options):
    """Calculate agreement for multi-select variables."""
    # Track metrics for each option
    option_metrics = {}
    kappa_values = []

    # Process each option separately
    for option in options:
        # Create binary matrix (1 if option selected, 0 if not)
        binary_matrix = agreement_matrix.applymap(
            lambda x: 1 if isinstance(x, list) and option in x else 0
        )

        # Calculate agreement for this option as binary choice
        option_agreement = calculate_binary_agreement(binary_matrix)
        option_metrics[option] = option_agreement
        kappa_values.append(option_agreement.get('kappa', 0))

    # Calculate overall metrics
    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0

    return {
        'overall_agreement': avg_kappa,
        'kappa_by_option': option_metrics,
        'average_kappa': avg_kappa
    }


def calculate_binary_agreement(binary_matrix):
    """Calculate agreement for binary (yes/no) choices."""
    # Get coder pairs
    coders = binary_matrix.columns
    n_coders = len(coders)

    # Track metrics
    total_agreements = 0
    total_comparisons = 0
    kappa_values = []

    # Compare each pair of coders
    for i in range(n_coders):
        for j in range(i + 1, n_coders):
            coder1 = coders[i]
            coder2 = coders[j]

            # Get rows where both coders provided answers
            pair_data = binary_matrix[[coder1, coder2]].dropna()

            if len(pair_data) == 0:
                continue

            # Count agreements (both 1 or both 0)
            agreements = (pair_data[coder1] == pair_data[coder2]).sum()
            total_agreements += agreements
            total_comparisons += len(pair_data)

            # Calculate Cohen's kappa
            observed_agreement = agreements / len(pair_data)

            # Calculate expected agreement for binary choice
            prob_coder1_yes = pair_data[coder1].mean()
            prob_coder1_no = 1 - prob_coder1_yes
            prob_coder2_yes = pair_data[coder2].mean()
            prob_coder2_no = 1 - prob_coder2_yes

            expected_agreement = (prob_coder1_yes * prob_coder2_yes) + (prob_coder1_no * prob_coder2_no)

            # Compute kappa if possible
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                kappa_values.append(kappa)

    # Calculate overall metrics
    overall_agreement = total_agreements / total_comparisons if total_comparisons > 0 else 0
    avg_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0

    return {
        'agreement_rate': overall_agreement
        # 'kappa': avg_kappa,
        # 'sample_size': total_comparisons
    }


def find_conflicts(agreement_matrix, variable, variable_info, variable_annotations):
    """Identify conflicts where coders disagree."""
    conflicts = []

    # Get mapping of task_id to platform_id
    platform_id_map = {}
    for _, row in variable_annotations.iterrows():
        if 'task_id' in row and 'platform_id' in row:
            platform_id_map[row['task_id']] = row['platform_id']

    # Check if we're using composite keys (for indexed variables)
    has_composite_key = False
    if len(agreement_matrix.index) > 0:
        first_idx = agreement_matrix.index[0]
        if isinstance(first_idx, str) and '_' in first_idx:
            has_composite_key = True

    var_type = variable_info.get('type', '')

    # Process each row in the agreement matrix
    for task_id, row in agreement_matrix.iterrows():
        values = row.dropna().tolist()
        coders = row.dropna().index.tolist()

        if len(values) < 2:
            continue  # Need at least 2 annotations to have conflict

        # Handle different variable types
        if var_type in ['single', 'single_select']:
            # Check if all values are the same
            if len(set(str(val) for val in values)) == 1:
                continue  # No conflict

            # Extract actual task_id and image_idx from composite key if needed
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

            # Record conflict
            conflict = {
                'task_id': actual_task_id,
                'platform_id': platform_id_map.get(actual_task_id),
                'variable': variable,
                'agreement_score': 0.0,  # Known disagreement
                'annotations': [
                    {'user_id': coders[i], 'value': values[i]}
                    for i in range(len(coders))
                ]
            }

            # Add image_idx if available
            if image_idx is not None:
                conflict['image_idx'] = image_idx

            conflicts.append(conflict)

        elif var_type in ['multi', 'multi_select']:
            # Compare lists of options
            serialized_values = [str(sorted(v) if isinstance(v, list) else v) for v in values]
            if len(set(serialized_values)) == 1:
                continue  # No conflict

            # Calculate option-by-option agreement
            options = variable_info.get('options', [])
            option_agreements = []

            for option in options:
                # Check if coders agree on including/excluding this option
                option_values = [1 if isinstance(v, list) and option in v else 0 for v in values]
                option_agreement = 1.0 if len(set(option_values)) == 1 else 0.0
                option_agreements.append(option_agreement)

            avg_agreement = sum(option_agreements) / len(option_agreements) if option_agreements else 0

            # Extract actual task_id and image_idx from composite key if needed
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

            # Record conflict
            conflict = {
                'task_id': actual_task_id,
                'platform_id': platform_id_map.get(actual_task_id),
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


def fix_users(df, usermap):
    """Map user IDs according to the provided mapping."""
    df['user_id'] = df['user_id'].map(usermap)
    return df
