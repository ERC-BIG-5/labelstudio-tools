from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ls_helper.models.result_models import ProjectResult


class ResultTransform:

    def __init__(self, res: "ProjectResult"):
        self.res = res

    def fulvia_output_format(self):
        """
        Transform annotation data from long format to binary-encoded format.
        Creates two DataFrames - one for each annotator across all tasks.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with columns: task_id, ann_id, variable, idx, value
            where 'value' contains lists of strings
        variable_info : dict
            Dictionary with variable metadata:
            {
                'variable_name': {
                    'type': 'single' or 'multiple',
                    'options': ['option1', 'option2', ...]
                }
            }

        Returns:
        --------
        tuple of (df_annotator1, df_annotator2)
            Two dataframes, one for first annotator per task, one for second annotator per task
        """

        # Ensure value column contains lists
        # if not df['value'].apply(lambda x: isinstance(x, list)).all():
        #     df = df.copy()
        #     df['value'] = df['value'].apply(lambda x: x if isinstance(x, list) else [x])

        # Explode the value lists to create one row per value
        df, ass = self.res.build_annotation_df(test_rebuild=True, ignore_groups=True)
        # df = res.raw_annotation_df

        df = self.res.limit_annotators_from_raw(df)
        df = self.res.filter_choices_only_from_raw(df)
        df = self.res.drop_variables_from_raw(df)
        variables = self.res.project_data.choices(exclude=["coding-game", "harmful_any", "harmful_visual"])
        variable_info = {
            v: {"options": v_d.options, "type": v_d.choice}
            for v, v_d in variables.items()
        }

        df_exploded = df.explode('value').reset_index(drop=True)

        # Create variable_option column
        df_exploded['variable_option'] = df_exploded['variable'] + '@' + df_exploded['value'].astype(str)

        # Create binary indicator
        df_exploded['indicator'] = 1

        # duplicate_check = df_exploded.groupby(['task_id', 'ann_id', 'variable_option']).size()
        # if (duplicate_check > 1).any():
        #     duplicates = duplicate_check[duplicate_check > 1]
        #     raise ValueError(f"Found duplicate entries: {duplicates.to_dict()}")

        platform_ids = df[['task_id', 'platform_id']].drop_duplicates()
        user_ids = df[['ann_id', 'user_id']].drop_duplicates()
        # Pivot to get binary matrix
        pivot_df = df_exploded.pivot(
            index=['task_id', 'ann_id'],
            columns='variable_option',
            values='indicator'
        ).fillna(0).astype(int)

        # Create all possible columns
        all_columns = []
        for var_name, var_meta in variable_info.items():
            for option in var_meta['options']:
                all_columns.append(f"{var_name}@{option}")

        # Find missing columns and add them all at once
        missing_columns = [col for col in all_columns if col not in pivot_df.columns]

        if missing_columns:
            # Create DataFrame with missing columns filled with 0s
            missing_df = pd.DataFrame(0,
                                      index=pivot_df.index,
                                      columns=missing_columns)
            # Concatenate all at once to avoid fragmentation
            pivot_df = pd.concat([pivot_df, missing_df], axis=1)

        # Select and sort columns
        column_order = []
        for var_name, var_options in variable_info.items():
            column_order.extend([f"{var_name}@{option}" for option in var_options['options']])
        pivot_df = pivot_df[sorted(all_columns, key=lambda x: column_order.index(x) if x in column_order else -1)]

        # Determine annotator pairs for each task
        task_ann_pairs = {}
        for task_id in pivot_df.index.get_level_values('task_id').unique():
            anns = pivot_df.loc[task_id].index.tolist()
            if len(anns) == 2:
                task_ann_pairs[task_id] = sorted(anns)  # Sort to ensure consistent ordering
            else:
                raise ValueError(f"Task {task_id} does not have exactly 2 annotators")

        # Create two separate DataFrames
        annotator1_data = []
        annotator2_data = []
        task_ids = []

        for task_id in sorted(task_ann_pairs.keys()):
            ann1, ann2 = task_ann_pairs[task_id]

            # Get data for each annotator
            annotator1_data.append(pivot_df.loc[(task_id, ann1)])
            annotator2_data.append(pivot_df.loc[(task_id, ann2)])
            task_ids.append(task_id)

        # Create final DataFrames
        df_annotator1 = pd.DataFrame(annotator1_data, index=task_ids)
        df_annotator1.index.name = 'task_id'

        df_annotator2 = pd.DataFrame(annotator2_data, index=task_ids)
        df_annotator2.index.name = 'task_id'

        return df_annotator1, df_annotator2
