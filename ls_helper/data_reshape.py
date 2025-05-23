from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from ls_helper.models.result_models import ProjectResult


class ResultTransform:

    def __init__(self, res: "ProjectResult"):
        self.res = res

    def fulvia_output_format(self) -> tuple[DataFrame, DataFrame]:
        """
        The Fulvia-format:
            - two dataframes, one for annotator1 and one for annotator2
            - each dataframe has one row per task
            - each row has a binary indicator for each variable-option combination

        :return: tuple[DataFrame, DataFrame]
        """

        # Explode the value lists to create one row per value
        # todo, maybe this could be stored in a specific file...
        df_orig, ass = self.res.build_annotation_df(test_rebuild=True, ignore_groups=True)

        df = self.res.limit_annotators_from_raw(df_orig)
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
            # Create a DataFrame with missing columns filled with 0s
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
        ann1_ids = []
        ann2_ids = []

        for task_id in sorted(task_ann_pairs.keys()):
            ann1, ann2 = task_ann_pairs[task_id]

            # Get data for each annotator
            annotator1_data.append(pivot_df.loc[(task_id, ann1)])
            annotator2_data.append(pivot_df.loc[(task_id, ann2)])
            task_ids.append(task_id)
            ann1_ids.append(ann1)
            ann2_ids.append(ann2)

        # Create final DataFrames
        df_annotator1 = pd.DataFrame(annotator1_data, index=task_ids)
        df_annotator1.index.name = 'task_id'
        df_annotator1['ann_id'] = ann1_ids

        df_annotator2 = pd.DataFrame(annotator2_data, index=task_ids)
        df_annotator2.index.name = 'task_id'
        df_annotator2['ann_id'] = ann2_ids

        p_ids = df_orig[['task_id', 'platform_id']].drop_duplicates().set_index('task_id')
        u_ids = df_orig[['ann_id', 'user_id']].drop_duplicates().set_index('ann_id')

        df_annotator1 = df_annotator1.join(p_ids)
        df_annotator2 = df_annotator2.join(p_ids)

        df_annotator1 = df_annotator1.reset_index().set_index('ann_id').join(u_ids).reset_index().set_index('task_id')
        df_annotator2 = df_annotator2.reset_index().set_index('ann_id').join(u_ids).reset_index().set_index('task_id')

        return df_annotator1, df_annotator2
