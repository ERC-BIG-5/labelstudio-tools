import re
from pathlib import Path

from pandas import DataFrame
from tqdm import tqdm

from ls_helper.models.variable_models import ChoiceVariableModel, NO_SINGLE_CHOICE
from tools.project_logging import get_logger

logger = get_logger(__file__)

from typing import Optional, List, Dict, Union, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
import csv
from irrCAC.raw import CAC


# Pydantic Models for Agreement Metrics
class OptionAgreement(BaseModel):
    """Model for option-level agreement metrics"""
    error: Optional[str] = None
    kappa: float = 1.0
    gwet: float = 1.0
    percent_agreement: float = 1.0
    number: int = 0  # For single-choice: number of rows with this option
    agreement_count: int = 0
    disagreement_count: int = 0
    total_selections: Optional[int] = None  # For single-choice
    # For multi-select
    total_rows_selected: Optional[int] = None
    total_tasks_analyzed: Optional[int] = None
    all_present_count: Optional[int] = None
    all_absent_count: Optional[int] = None
    conflicts: Optional[List[str]] = None


class AnnotationInfo(BaseModel):
    """Model for annotation information in conflicts"""
    user_id: str | int
    value: str
    option: Optional[str] = None


class ConflictInfo(BaseModel):
    """Model for conflict information"""
    task_id: int
    platform_id: Optional[str] = None
    variable: Optional[str] = None
    option: Optional[str] = None
    agreement_score: float = 0.0
    annotations: List[AnnotationInfo]
    image_idx: Optional[int] = None
    conflict_type: Optional[str] = None


class SingleChoiceAgreement(BaseModel):
    """Model for single-choice variable agreement results"""
    kappa: float
    gwet: float
    percent_agreement: float
    total_rows: int
    agreement_count: int
    disagreement_count: int
    counts: Dict[str, int]
    option_results: Dict[str, OptionAgreement]
    conflicts: Optional[List[str]] = None


class MultiChoiceAgreement(BaseModel):
    """Model for multi-choice variable agreement results"""
    counts: Dict[str, int]
    option_results: Dict[str, OptionAgreement]
    conflicts: List[ConflictInfo]


class VariableTypeAgreement(BaseModel):
    """Model for agreement metrics by variable type"""
    average_kappa: float
    average_gwet: float
    variables: Dict[str, Union[SingleChoiceAgreement, MultiChoiceAgreement]]
    conflict_rate: float
    conflict_count: int


class OverallAgreement(BaseModel):
    """Model for overall agreement metrics"""
    kappa: float
    gwet: float


class AgreementMetrics(BaseModel):
    """Model for all agreement metrics"""
    overall: OverallAgreement
    single_choice: VariableTypeAgreement
    multiple_choice: VariableTypeAgreement

    @property
    def all_variables(self) -> dict[str, Union[SingleChoiceAgreement, MultiChoiceAgreement]]:
        return self.single_choice.variables | self.multiple_choice.variables


class SummaryStats(BaseModel):
    """Model for summary statistics"""
    total_tasks: int
    total_coders: int
    total_variables: int
    total_annotations: int
    total_conflicts: int
    single_choice_variables: int
    multiple_choice_variables: int
    indexed_variables_count: int
    base_variables_count: int
    conflict_rate: float


class AgreementReport(BaseModel):
    """Model for the complete agreement report"""
    summary_stats: SummaryStats
    agreement_metrics: AgreementMetrics
    conflicts: List[ConflictInfo]


# CSV Export Model for flattened representation
class AgreementCsvRow(BaseModel):
    """Model for a row in the agreement metrics CSV"""
    variable_type: str
    variable: str
    option: str = ""
    kappa: Optional[float] = None
    gwet: Optional[float] = None
    percent_agreement: Optional[float] = None
    total_tasks: Optional[int] = None
    agreement_count: Optional[int] = None
    disagreement_count: Optional[int] = None
    option_selected_count: Optional[int] = None
    all_present_count: Optional[int] = None
    all_absent_count: Optional[int] = None


def _calculate_single_select_agreement2(agreement_matrix,
                                        variable_info,
                                        counts: bool = True,
                                        collect_conflicts_agreements: bool = True) -> SingleChoiceAgreement:
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
    # Round to 4 digits
    percent_agreement = round(percent_agreement, 4)

    all_conflicts = []

    # Calculate Fleiss' kappa
    try:
        cac = CAC(agreement_matrix)
        fk = cac.fleiss()
        kappa = fk["est"]["coefficient_value"]
        gwet = cac.gwet()["est"]["coefficient_value"]
        # Round to 4 digits
        kappa = round(kappa, 4)
        gwet = round(gwet, 4)
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
    # conflicts = None
    # if collect_conflicts_agreements:
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

        # Calculate option-level percent agreement
        option_percent_agreement = agreement_on_option_count / option_row_count if option_row_count > 0 else 1.0
        # Round to 4 digits
        option_percent_agreement = round(option_percent_agreement, 4)

        error = None
        # Option-specific kappa
        try:
            if len(option_rows) > 0:
                cac = CAC(option_rows)
                option_kappa = cac.fleiss()["est"]["coefficient_value"]
                option_gwet = cac.gwet()["est"]["coefficient_value"]
                # Round to 4 digits
                option_kappa = round(option_kappa, 4)
                option_gwet = round(option_gwet, 4)
            else:
                option_kappa = 1.0
                option_gwet = 1.0
        except Exception as e:
            option_kappa = 1.0 if disagreement_on_option_count == 0 else 0.0
            option_gwet = 1.0
            error = str(e)

        # Use Pydantic model for option results
        option_results[option] = OptionAgreement(
            error=error,
            kappa=float(option_kappa),
            gwet=float(option_gwet),
            percent_agreement=float(option_percent_agreement),
            number=int(option_row_count) if hasattr(option_row_count, 'item') else option_row_count,
            agreement_count=int(agreement_on_option_count) if hasattr(agreement_on_option_count,
                                                                      'item') else agreement_on_option_count,
            disagreement_count=int(disagreement_on_option_count) if hasattr(disagreement_on_option_count,
                                                                            'item') else disagreement_on_option_count,
            total_selections=option_counts[option]
        )

    # Use Pydantic model for the result
    return SingleChoiceAgreement(
        kappa=float(kappa) if hasattr(kappa, 'item') else kappa,
        gwet=float(gwet) if hasattr(gwet, 'item') else gwet,
        percent_agreement=float(percent_agreement) if hasattr(percent_agreement, 'item') else percent_agreement,
        total_rows=int(total_valid_rows) if hasattr(total_valid_rows, 'item') else total_valid_rows,
        agreement_count=int(agreement_count) if hasattr(agreement_count, 'item') else agreement_count,
        disagreement_count=int(disagreement_count) if hasattr(disagreement_count, 'item') else disagreement_count,
        counts=option_counts,
        option_results=option_results,
        conflicts=all_conflicts
    )


def _calculate_multi_select_agreement(agreement_matrix, variable_info,
                                      counts: bool = True,
                                      collect_conflicts_agreements: bool = True) -> MultiChoiceAgreement:
    """
    Calculate agreement for multi-select variables by creating
    binary matrices for each option.
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
        # SIMPLEST FIX: Use the map function with a safe lambda that handles NaN values correctly
        binary_matrix = agreement_matrix.map(
            lambda x: 1 if isinstance(x, list) and option in x else (
                0 if isinstance(x, list) else np.nan
            )
        )

        # Only look at rows with at least 2 annotators
        multiple_annotator_mask = binary_matrix.count(axis=1) >= 2
        rows_with_multiple_annotators = binary_matrix[multiple_annotator_mask]

        # Skip the rest of processing if no rows have multiple annotators
        if len(rows_with_multiple_annotators) == 0:
            option_results[option] = OptionAgreement(
                error=None,
                kappa=1.0,
                gwet=1.0,
                percent_agreement=1.0,
                total_rows_selected=0,
                total_tasks_analyzed=0,
                agreement_count=0,
                disagreement_count=0,
                all_present_count=0,
                all_absent_count=0,
                conflicts=None
            )
            continue

        # Count rows where all coders agreed this option was present
        all_present_mask = rows_with_multiple_annotators.apply(
            lambda row: row.dropna().nunique() == 1 and row.dropna().iloc[0] == 1,
            axis=1
        )
        all_present_count = all_present_mask.sum()

        # Count rows where all coders agreed this option was absent
        all_absent_mask = rows_with_multiple_annotators.apply(
            lambda row: row.dropna().nunique() == 1 and row.dropna().iloc[0] == 0,
            axis=1
        )
        all_absent_count = all_absent_mask.sum()

        # Total agreement count (all present OR all absent)
        agreement_count = all_present_count + all_absent_count

        # Calculate disagreements (rows where some coders say present and others say absent)
        disagreement_mask = rows_with_multiple_annotators.apply(
            lambda row: row.dropna().nunique() > 1,  # Must have both 0 and 1 in the row
            axis=1
        )
        disagreement_rows = rows_with_multiple_annotators[disagreement_mask]
        disagreement_count = len(disagreement_rows)

        # Rows where this option was selected at least once
        rows_with_option = rows_with_multiple_annotators[rows_with_multiple_annotators.eq(1).any(axis=1)]
        option_presence_count = len(rows_with_option)

        # Calculate percent agreement based on rows with multiple annotators
        total_analyzed = len(rows_with_multiple_annotators)
        percent_agreement = agreement_count / total_analyzed if total_analyzed > 0 else 1.0
        # Round to 4 digits
        percent_agreement = round(percent_agreement, 4)

        # Calculate kappa and gwet for this option
        kappa = 1.0
        gwet = 1.0
        error = None

        try:
            if len(rows_with_multiple_annotators) > 0:
                cac = CAC(rows_with_multiple_annotators)
                kappa = cac.fleiss()["est"]["coefficient_value"]
                gwet = cac.gwet()["est"]["coefficient_value"]
                # Round to 4 digits
                kappa = round(kappa, 4)
                gwet = round(gwet, 4)
        except Exception as e:
            error = str(e)
            logger.error(f"Agreement calculation error for option {option}: {error}")

        # Get conflicts for this option if needed
        option_conflicts = []
        if collect_conflicts_agreements:
            # Only collect ACTUAL disagreements (where some coders say present and others say absent)
            for task_id in disagreement_rows.index:
                row_data = binary_matrix.loc[task_id].dropna()

                # Check if there's a mix of 0s and 1s (true disagreement)
                values_set = set(row_data)
                if len(values_set) > 1:  # Must have both 0 and 1 to be a true disagreement
                    option_conflicts.append(task_id)

                    # Create annotations list using Pydantic model
                    annotations = [
                        AnnotationInfo(
                            user_id=coder,
                            value='present' if value == 1 else 'absent'
                        )
                        for coder, value in row_data.items()
                    ]

                    # Create conflict info using Pydantic model
                    conflict_info = ConflictInfo(
                        task_id=task_id,
                        option=option,
                        agreement_score=0,
                        annotations=annotations,
                        conflict_type='multiple_choice'
                    )
                    all_conflicts.append(conflict_info)

        # Store metrics for this option using Pydantic model
        option_results[option] = OptionAgreement(
            error=error,
            kappa=float(kappa),
            gwet=float(gwet),
            percent_agreement=float(percent_agreement),
            total_rows_selected=int(option_presence_count),
            total_tasks_analyzed=int(total_analyzed),
            agreement_count=int(agreement_count),
            disagreement_count=int(disagreement_count),
            all_present_count=int(all_present_count),
            all_absent_count=int(all_absent_count),
            conflicts=[str(c) for c in option_conflicts] if option_conflicts else None
        )

    # Final result using Pydantic model
    return MultiChoiceAgreement(
        counts=option_counts,
        option_results=option_results,
        conflicts=all_conflicts
    )


def export_agreement_metrics_to_csv(agreement_report: AgreementReport, output_file: Path) -> list[dict]:
    """
    Export agreement metrics to a CSV file using Pydantic models.

    Parameters:
    -----------
    agreement_report : AgreementReport
        The agreement report returned by analyze_coder_agreement

    output_file : str
        Path to the output CSV file
    """
    # Prepare data for CSV using our CSV model
    rows = []

    # Process single-choice variables
    single_choice_vars = agreement_report.agreement_metrics.single_choice.variables
    for var_name, var_data in single_choice_vars.items():
        # Basic variable info
        row = AgreementCsvRow(
            variable_type="single_choice",
            variable=var_name,
            option="VARIABLE_LEVEL",
            kappa=var_data.kappa,
            gwet=var_data.gwet,
            percent_agreement=var_data.percent_agreement,
            total_tasks=var_data.total_rows,
            agreement_count=var_data.agreement_count,
            disagreement_count=var_data.disagreement_count
        )
        rows.append(row)

        # Add option-level data for single choice
        if var_data.option_results:
            for opt_name, opt_data in var_data.option_results.items():
                opt_row = AgreementCsvRow(
                    variable_type="single_choice",
                    variable=var_name,
                    option=opt_name,
                    kappa=opt_data.kappa,
                    gwet=opt_data.gwet,
                    percent_agreement=opt_data.percent_agreement,
                    total_tasks=opt_data.number,
                    agreement_count=opt_data.agreement_count,
                    disagreement_count=opt_data.disagreement_count,
                    option_selected_count=opt_data.total_selections
                )
                rows.append(opt_row)

    # Process multiple-choice variables
    multiple_choice_vars = agreement_report.agreement_metrics.multiple_choice.variables
    for var_name, var_data in multiple_choice_vars.items():
        # Add option-level data
        if var_data.option_results:
            for opt_name, opt_data in var_data.option_results.items():
                opt_row = AgreementCsvRow(
                    variable_type="multiple_choice",
                    variable=var_name,
                    option=opt_name,
                    kappa=opt_data.kappa,
                    gwet=opt_data.gwet,
                    percent_agreement=opt_data.percent_agreement,
                    total_tasks=opt_data.total_tasks_analyzed,
                    agreement_count=opt_data.agreement_count,
                    disagreement_count=opt_data.disagreement_count,
                    option_selected_count=opt_data.total_rows_selected,
                    all_present_count=opt_data.all_present_count,
                    all_absent_count=opt_data.all_absent_count
                )
                rows.append(opt_row)

    # Add summary rows
    rows.append(AgreementCsvRow(
        variable_type="SUMMARY",
        variable="ALL_SINGLE_CHOICE",
        kappa=round(agreement_report.agreement_metrics.single_choice.average_kappa, 4),
        gwet=round(agreement_report.agreement_metrics.single_choice.average_gwet, 4),
        disagreement_count=agreement_report.agreement_metrics.single_choice.conflict_count
    ))

    rows.append(AgreementCsvRow(
        variable_type="SUMMARY",
        variable="ALL_MULTIPLE_CHOICE",
        kappa=round(agreement_report.agreement_metrics.multiple_choice.average_kappa, 4),
        gwet=round(agreement_report.agreement_metrics.multiple_choice.average_gwet, 4),
        disagreement_count=agreement_report.agreement_metrics.multiple_choice.conflict_count
    ))

    # Add overall summary
    rows.append(AgreementCsvRow(
        variable_type="SUMMARY",
        variable="OVERALL",
        kappa=round(agreement_report.agreement_metrics.overall.kappa, 4),
        gwet=round(agreement_report.agreement_metrics.overall.gwet, 4),
        total_tasks=agreement_report.summary_stats.total_tasks,
        disagreement_count=agreement_report.summary_stats.total_conflicts
    ))

    # Directly use model's fields for CSV headers
    fieldnames = list(AgreementCsvRow.__annotations__.keys())

    # Write to CSV
    with output_file.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Convert Pydantic models to dictionaries for CSV writing
        writer.writerows([row.model_dump() for row in rows])
    return rows


def analyze_coder_agreement(raw_annotations, assignments, choices,
                            min_coders: Optional[int] = 2,
                            field_names: Optional[List[str]] = None) -> AgreementReport:
    """
    End-to-end function to analyze coder agreement across annotations.
    Modified to treat indexed variables as part of a unified dataset.
    """
    # Step 0: Filter out tasks with less than 'min_coders' coders
    initial_task_count = raw_annotations['task_id'].nunique()
    task_annotation_counts = raw_annotations.groupby('task_id')['ann_id'].nunique()
    tasks_with_multiple_anns = task_annotation_counts[task_annotation_counts >= min_coders].index
    filtered_df = raw_annotations[raw_annotations['task_id'].isin(tasks_with_multiple_anns)]
    remaining_task_count = filtered_df['task_id'].nunique()
    print(f"Initial tasks: {initial_task_count}, Remaining tasks: {remaining_task_count}, "
          f"Removed tasks: {initial_task_count - remaining_task_count}")

    # Step 1: Create the assignment tracking DataFrame
    assignments_df = _create_assignment_tracking(assignments)

    # Step 2: Create annotations DataFrame without defaults
    annotations_df = create_annotations_dataframe(filtered_df)

    # Add image index information and create task_key
    base_to_indexed = add_image_index_column(annotations_df)
    print(f"{base_to_indexed=}")

    # Ensure task_key exists for all annotations
    if 'task_key' not in annotations_df.columns:
        annotations_df['task_key'] = (
                annotations_df['task_id'].astype(str) + '_' +
                annotations_df['image_idx'].astype(str)
        )

    # Initialize results containers
    all_variable_agreements = {}
    all_conflicts = []

    # PREPROCESSING STEP: Categorize variables to eliminate redundant checks
    variable_categories = {
        'base_variables': set(),  # Base variables with no indexed versions
        'indexed_bases': set(),  # Base variables that have indexed versions
        'process_with_base': {},  # Maps indexed vars to their base for processing together
        'skip_processing': set()  # Variables to skip (indices > 0)
    }

    # Categorize all variables upfront to avoid redundant checks
    for variable in choices.keys():
        # Check if this is a base variable with indexed versions
        if variable in base_to_indexed:
            variable_categories['indexed_bases'].add(variable)
            # Mark higher indices to be skipped
            for idx in base_to_indexed[variable]:
                if idx > 0:
                    indexed_var = f"{variable}_{idx}"
                    variable_categories['skip_processing'].add(indexed_var)
                    variable_categories['process_with_base'][indexed_var] = variable
            # Map the _0 version to the base
            zero_indexed_var = f"{variable}_0"
            if zero_indexed_var in choices:
                variable_categories['process_with_base'][zero_indexed_var] = variable
        else:
            # Check if this might be an indexed variable
            match = re.match(r'^(.+)_(\d+)$', variable)
            if match:
                base_name, idx = match.groups()
                idx = int(idx)
                if base_name in variable_categories['indexed_bases']:
                    if idx > 0:
                        variable_categories['skip_processing'].add(variable)
                    variable_categories['process_with_base'][variable] = base_name
                else:
                    # It's a variable with _number suffix but not part of indexed series
                    variable_categories['base_variables'].add(variable)
            else:
                variable_categories['base_variables'].add(variable)

    print(f"Base variables: {len(variable_categories['base_variables'])}")
    print(f"Indexed base variables: {len(variable_categories['indexed_bases'])}")
    print(f"Skip processing: {len(variable_categories['skip_processing'])}")
    print(f"Process with base: {len(variable_categories['process_with_base'])}")

    indexed_variable_map = {}  # Maps base variable names to True if they have indexed versions
    processed_base_vars = set()  # Tracks which base variables have already been processed
    indexed_vars_to_skip = set()  # Variables to skip (indices > 0)
    base_vars_map = {}  # Maps indexed _0 variables to their base name

    # Create a mapping to track variables with indices
    for base_name, indexed_vars in base_to_indexed.items():
        if len(indexed_vars) > 0:
            # Mark this as a base variable with indexed versions
            indexed_variable_map[base_name] = True
            print(f"Adding {base_name} to indexed_variable_map")

            # Create a list of indexed variables to skip (only indices > 0)
            higher_indexed_vars = [f"{base_name}_{idx}" for idx in indexed_vars if idx > 0]
            indexed_vars_to_skip.update(higher_indexed_vars)

            # Map the _0 version to the base name
            zero_indexed_var = f"{base_name}_0"
            base_vars_map[zero_indexed_var] = base_name

    print(f"Indexed base variables: {list(indexed_variable_map.keys())}")
    print(f"Indexed variables to skip: {list(indexed_vars_to_skip)}")
    print(f"Base vars map: {base_vars_map}")

    # Process variables based on their categorization
    for variable, variable_info in tqdm(choices.items()):
        if field_names and variable not in field_names:
            continue

        print(f"Processing variable: {variable}")
        print(f"Is this an indexed base? {variable in indexed_variable_map}")

        # Check if this is a base variable WITH indexed versions
        if variable in indexed_variable_map:
            print(f"Processing {variable} as a base variable with indexed versions")

            # Skip if we've already processed this base variable
            if variable in processed_base_vars:
                print(f"Already processed base variable {variable}, skipping")
                continue

            processed_base_vars.add(variable)

            # Find all variables with this base name - use the pattern once
            base_pattern = f"^{re.escape(variable)}_\\d+"
            indexed_vars = [v for v in annotations_df['variable'].unique()
                            if re.match(base_pattern, str(v))]

            print(f"Found indexed versions: {indexed_vars}")

            # Get all annotations for indexed versions
            indexed_annotations = annotations_df[annotations_df['variable'].isin(indexed_vars)]

            if len(indexed_annotations) > 0:
                print(f"Found {len(indexed_annotations)} annotations for indexed versions of {variable}")

                # Apply defaults using our improved function for ALL indexed versions together
                variable_with_defaults = apply_defaults_for_variable(
                    indexed_annotations,
                    assignments_df,
                    variable,  # Use base variable name
                    variable_info
                )

                # Create agreement matrix using task_key
                agreement_matrix = variable_with_defaults.pivot(
                    index='task_key',
                    columns='user_id',
                    values='value'
                )

                # Calculate agreement
                variable_agreement = calculate_agreement(
                    agreement_matrix,
                    variable_info
                )

                # Identify conflicts
                variable_conflicts = identify_conflicts(
                    agreement_matrix,
                    variable,
                    variable_info,
                    variable_with_defaults
                )

                # Store results
                all_variable_agreements[variable] = variable_agreement
                all_conflicts.extend(variable_conflicts)

                # Continue to next variable without individual processing
                continue

        if variable in indexed_vars_to_skip:
            print(f"Skipping indexed variable: {variable}")
            continue

        # For _0 indexed variables, we'll process them but store under the base name
        result_var_name = variable
        if variable in base_vars_map:
            print(f"Processing {variable} as base variable {base_vars_map[variable]}")
            result_var_name = base_vars_map[variable]

        # Get all annotations for this variable
        variable_annotations = annotations_df[annotations_df['variable'] == variable]

        if len(variable_annotations) == 0:
            # Skip variables with no data
            print(f"No data on {variable}")
            continue

        # Standard processing for regular variables
        variable_with_defaults = apply_defaults_for_variable(
            variable_annotations,
            assignments_df,
            variable,
            variable_info
        )

        # Create agreement matrix using task_key
        agreement_matrix = variable_with_defaults.pivot(
            index='task_key',
            columns='user_id',
            values='value'
        )

        # For single-choice, clear the agreement matrix
        if variable_info.choice == 'single':
            cleared_agreement_matrix = clear_agreement_matrix(agreement_matrix, variable_info)
            variable_agreement = calculate_agreement(
                cleared_agreement_matrix,
                variable_info
            )
        else:  # multi_select
            variable_agreement = calculate_agreement(
                agreement_matrix,
                variable_info
            )

        # Store results
        all_variable_agreements[result_var_name] = variable_agreement

        # Identify conflicts
        variable_conflicts = identify_conflicts(
            agreement_matrix,
            variable,
            variable_info,
            variable_with_defaults
        )

        all_conflicts.extend(variable_conflicts)

    # Generate the final report
    agreement_report_dict = generate_agreement_report(
        all_variable_agreements,
        all_conflicts,
        choices,
        annotations_df,
        base_to_indexed,
        variable_categories['skip_processing']  # Pass our clean skip set
    )

    # Convert to Pydantic model
    return AgreementReport.model_validate(agreement_report_dict)


def identify_conflicts(agreement_matrix: DataFrame, variable: str, variable_info: ChoiceVariableModel,
                       variable_annotations):
    """
    Identify conflicts for this variable using task_key.
    """
    conflicts = []

    # Get mapping of task_key to task_id and platform_id
    task_key_to_info = {}
    for _, row in variable_annotations.iterrows():
        if 'task_key' in row and 'task_id' in row:
            # Store mapping from task_key to task_id and platform_id
            task_key_to_info[row['task_key']] = {
                'task_id': row['task_id'],
                'platform_id': str(row.get('platform_id', '')),
                'image_idx': row.get('image_idx', 0)
            }

    # Different handling for single vs multiple choice variables
    if variable_info.choice == 'single':
        # For single-select, identify conflicts where annotators chose different options
        for task_key, row in agreement_matrix.iterrows():
            # Get task info
            task_info = task_key_to_info.get(task_key, {})
            task_id = task_info.get('task_id', int(task_key.split('_')[0]) if '_' in task_key else task_key)
            platform_id = task_info.get('platform_id')
            image_idx = task_info.get('image_idx', 0)

            # Drop NaN values and get values and coders
            values = row.dropna().tolist()
            coders = row.dropna().index.tolist()

            if len(values) < 2:
                continue  # Need at least 2 annotations to have a conflict

            # For single-choice, extract first element from lists
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

            # Convert numeric indices to option labels for readability
            labeled_annotations = []
            for i, value in enumerate(processed_values):
                if isinstance(value, (int, float)) and not np.isnan(value):
                    try:
                        # Convert index to label
                        label = variable_info.options[int(value)]
                    except (IndexError, ValueError):
                        label = str(value) + " (unknown option)"
                else:
                    label = str(value)

                labeled_annotations.append(AnnotationInfo(
                    user_id=coders[i],
                    value=label
                ))

            # Record conflict
            conflict = ConflictInfo(
                task_id=int(task_id) if isinstance(task_id, str) and task_id.isdigit() else task_id,
                platform_id=platform_id if platform_id else None,
                variable=variable,
                agreement_score=0.0,
                annotations=labeled_annotations,
                conflict_type='single_choice'
            )

            # Add image_idx if available
            if image_idx is not None and image_idx != 0:
                conflict.image_idx = image_idx

            conflicts.append(conflict)

    else:  # Multiple choice
        # For multi-select, check conflicts option by option
        options = variable_info.options

        for option in options:
            # Create binary matrix for this option (1 if present, 0 if absent)
            binary_matrix = agreement_matrix.map(
                lambda x: 1 if isinstance(x, list) and option in x else (
                    0 if isinstance(x, list) else np.nan
                )
            )

            # Look for disagreements on this option (must have both 0s and 1s)
            for task_key, row in binary_matrix.iterrows():
                # Get task info
                task_info = task_key_to_info.get(task_key, {})
                task_id = task_info.get('task_id', int(task_key.split('_')[0]) if '_' in task_key else task_key)
                platform_id = task_info.get('platform_id')
                image_idx = task_info.get('image_idx', 0)

                # Drop NaN values
                values = row.dropna()
                if len(values) < 2:
                    continue  # Need at least 2 annotations

                # Check if there's a true disagreement (must have both 0s and 1s)
                if set(values) != {0, 1}:
                    continue  # Not a true disagreement if all have same value

                # Create annotations for coders who actually coded
                option_annotations = []
                for coder, value in values.items():
                    option_annotations.append(AnnotationInfo(
                        user_id=coder,
                        value='present' if value == 1 else 'absent',
                        option=option
                    ))

                # Record conflict for this option
                conflict = ConflictInfo(
                    task_id=int(task_id) if isinstance(task_id, str) and task_id.isdigit() else task_id,
                    platform_id=platform_id if platform_id else None,
                    variable=variable,
                    option=option,
                    agreement_score=0.0,
                    annotations=option_annotations,
                    conflict_type='multiple_choice'
                )

                # Add image_idx if it's not the default value
                if image_idx is not None and image_idx != 0:
                    conflict.image_idx = image_idx

                conflicts.append(conflict)

    return conflicts


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


def generate_agreement_report(all_variable_agreements, all_conflicts, choices, annotations_df,
                              base_to_indexed=None, indexed_vars_to_skip=None):
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
    print("Variables received in generate_agreement_report:", sorted(all_variable_agreements.keys()))

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

    if indexed_vars_to_skip:
        print(f"Received {len(indexed_vars_to_skip)} indexed variables to skip")
        print(f"First 5 examples: {list(indexed_vars_to_skip)[:5]}")

    print("All variables:", sorted(all_variable_agreements.keys()))
    if indexed_vars_to_skip:
        print("Indexed variables to skip:", sorted(indexed_vars_to_skip))

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
        # Debug what variables are being processed
        print(f"Checking variable {var_name} for processing")

        if indexed_vars_to_skip and var_name in indexed_vars_to_skip:
            print(f"Skipping indexed variable: {var_name}")
            continue

        var_info = None
        # Try direct lookup first
        if var_name in choices:
            var_info = choices[var_name]
        else:
            # For base variables that were renamed from indexed_0, try looking up various forms
            # Check if this might be a base name for an indexed variable
            indexed_0_name = f"{var_name}_0"
            if indexed_0_name in choices:
                var_info = choices[indexed_0_name]

        if var_info is None:
            print(f"No variable info found for {var_name}, skipping")
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
    Create an agreement matrix using the composite task_key as the index.
    """
    # Use the task_key if it exists, otherwise create it
    if 'task_key' not in variable_annotations.columns:
        if 'image_idx' in variable_annotations.columns:
            variable_annotations['task_key'] = (
                    variable_annotations['task_id'].astype(str) + '_' +
                    variable_annotations['image_idx'].astype(str)
            )
        else:
            variable_annotations['task_key'] = variable_annotations['task_id'].astype(str) + '_0'

    # Pivot using the task_key as index
    agreement_matrix = variable_annotations.pivot(
        index='task_key',
        columns='user_id',
        values='value'
    )

    return agreement_matrix


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
    Also creates a composite task_key for agreement calculations.
    """
    # Extract base names and indices (existing code)
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

    # Create a composite task key that combines task_id and image_idx
    df['task_key'] = df['task_id'].astype(str) + '_' + df['image_idx'].astype(str)

    # Create mapping of base variables to their indices (existing code)
    base_to_indices = {}
    for base, idx in zip(base_names, indices):
        if idx > 0:  # Only track variables with indices
            if base not in base_to_indices:
                base_to_indices[base] = []
            base_to_indices[base].append(idx)

    return base_to_indices


def clear_agreement_matrix(agreement_matrix: DataFrame, variable_info: ChoiceVariableModel) -> DataFrame:
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
                                variable_info: ChoiceVariableModel):
    """
    Apply default values for a specific variable where annotations are missing.
    Uses task_key (task_id_image_idx) for handling indexed variables.
    """
    # Reset index on assignments to get task_id and user_id as columns
    assignments = assignments_df.reset_index()

    # Determine unique image indices for this variable
    indices = [0]  # Default for non-indexed variables
    if 'image_idx' in variable_annotations.columns:
        indices = sorted(variable_annotations['image_idx'].unique())

    # Create combinations for each task-coder-index
    all_combinations = []
    for _, row in assignments.iterrows():
        for idx in indices:
            combo = {
                'task_id': row['task_id'],
                'user_id': row['user_id'],
                'variable': variable,
                'image_idx': idx,
                'task_key': f"{row['task_id']}_{idx}",
                # Include metadata if available
                'ann_id': row.get('ann_id', None),
                'timestamp': row.get('ts', None),
                'platform_id': row.get('platform_id', None)
            }
            all_combinations.append(combo)

    combinations_df = pd.DataFrame(all_combinations)

    # Ensure variable_annotations has task_key
    if 'task_key' not in variable_annotations.columns:
        variable_annotations = variable_annotations.copy()
        variable_annotations['task_key'] = (
                variable_annotations['task_id'].astype(str) + '_' +
                variable_annotations.get('image_idx', 0).astype(str)
        )

    # Merge on task_key and user_id to properly handle indexed variables
    merged = pd.merge(
        combinations_df,
        variable_annotations,
        on=['task_key', 'user_id'],
        how='left',
        suffixes=('', '_existing')
    )

    # Clean up duplicate columns
    for col in merged.columns:
        if col.endswith('_existing'):
            base_col = col[:-9]
            # Keep existing values where available
            merged[base_col] = merged[col].combine_first(merged[base_col])
            merged = merged.drop(columns=[col])

    # Apply defaults for value column
    if 'value' not in merged.columns or merged['value'].isna().all():
        if variable_info.choice == 'single':
            merged['value'] = [variable_info.safe_default]
        else:  # multi_select
            merged['value'] = []
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
