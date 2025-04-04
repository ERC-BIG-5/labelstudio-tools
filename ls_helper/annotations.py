from typing import Any

import numpy as np
import pandas as pd
from deprecated.classic import deprecated
from pandas import DataFrame

from ls_helper.models.interface_models import InterfaceData, ProjectFieldsExtensions


def convert_strings_to_indices(df: DataFrame, string_list: list[str]):
    result_df = df.copy()

    # Create a mapping dictionary for faster lookups
    # Each string maps to its index in the list
    string_to_index = {s: i for i, s in enumerate(string_list)}

    # Define a function to apply to each element
    def map_to_index(value):
        if pd.isna(value):
            return np.nan
        elif value in string_to_index:
            return string_to_index[value]
        else:
            # Optional: handle case when string is not in the list
            # Could return np.nan, -1, or raise an error
            return np.nan

    # Apply the function to all elements in the DataFrame
    result_df = result_df.map(lambda x: map_to_index(x))

    # Convert all columns to int32, with NaN represented as pd.NA
    for col in result_df.columns:
        # First convert to nullable integer type
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        result_df[col] = result_df[col].astype('Int32')  # Note: capital 'I' for nullable integer

    return result_df

@deprecated(reason="somewhere else...")
def collect_variable_infos(annotation_structure: InterfaceData,
                           data_extensions: ProjectFieldsExtensions) -> dict[str,dict[str, Any]] :
    variables = {}

    fixes = data_extensions.extensions
    for var, fix_info in fixes.items():
        if new_name := fixes[var].name_fix:
            name = new_name
        else:
            name = var

        if name not in annotation_structure.ordered_fields:
            continue

        default = fix_info.default
        if name in annotation_structure.inputs:
            continue
        type = annotation_structure.question_type(name)
        if type in ["single", "multiple"]:
            options = annotation_structure.choices[name].raw_options_list()
        else:
            options = []
        variables[name] = {
            "name": name,
            "type": type,
            "options": options,
            "default": default
        }
    return variables