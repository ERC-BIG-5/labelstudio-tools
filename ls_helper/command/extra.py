from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer

from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS
from tools.project_logging import get_logger

logger = get_logger(__file__)

extras_app = typer.Typer(
    name="Extra things", pretty_exceptions_show_locals=True
)


@extras_app.command(short_help="[stats] Annotation basic results")
def get_confusions(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Argument()] = None,
        language: Annotated[Optional[str], typer.Argument()] = None,
        accepted_ann_age: Annotated[
            int, typer.Option(help="Download annotations if older than x hours")
        ] = 6,
        min_coders: Annotated[int, typer.Option()] = 2,
) -> Path:
    po = get_project(id, alias, platform, language)
    # po.validate_extensions()
    mp = po.get_annotations_results(accepted_ann_age=accepted_ann_age)
    df, _ = mp.get_annotation_df()

    ##############################
    # mp.flatten_annotation_results().to_csv("testing.csv")
    # df.to_csv("whatisthis3.csv")
    ##############################

    all_values = ['aesthetics',
                  'cultural-identity',
                  'good-life',
                  'kinship',
                  'literacy',
                  'livelihoods',
                  'personal-identity',
                  'reciprocity',
                  'sense-of-agency',
                  'sense-of-place',
                  'social-cohesion',
                  'social-memory',
                  'social-relations',
                  'social-responsibility',
                  'spirituality',
                  'stewardship-principle',
                  'stewardship-eudaimonia',
                  'well-being']

    # todo use something like this instead:
    # rel_value_option: ChoiceVariableModel = po.variables()["rel-value_text"]
    # all_values = rel_value_option.options

    # Helper functions
    def find_drv(data, item):
        if item in data:
            return True
        else:
            return False

    def clean_list_like(text):
        text = text[2:-2]
        if ", " in text:
            return text.split(", ")
        else:
            return [text]

    def get_proportion(clean, conf):
        if clean + conf == 0:
            return 0.00
        else:
            return round(conf / (clean + conf), 2)

    def transform_matrix(matrix):
        matrix_add = np.add(matrix, np.transpose(matrix))
        ind, matrix_end = 0, []
        for item in matrix_add:
            row, count = [], 0
            for pos in item:
                if count < ind:
                    row.append(pos)
                else:
                    row.append(0)
                count += 1
            matrix_end.append(row)
            ind += 1
        return np.transpose(matrix_end)

    # Prepare the output dataframes
    conf_all = pd.DataFrame(all_values, columns=["drv"])
    conf_text = conf_all.copy()
    conf_visual = conf_all.copy()
    # Produce temporary working dataframes
    df_text = df[(df["variable"] == "rel-value_text")]
    df_visual = df[(df["variable"] == "rel-value_visual")]
    # Get the count of annotations for every DRV
    count_drv_text, count_drv_visual = [], []
    for item in all_values:
        df_text["temp_count"] = df_text["value"].apply(find_drv, args=(item,))
        count_drv_text.append(df_text[df_text["temp_count"] == True].shape[0])
        df_text["temp_count"] = False
        df_visual["temp_count"] = df_visual["value"].apply(find_drv, args=(item,))
        count_drv_visual.append(df_visual[df_visual["temp_count"] == True].shape[0])
        df_visual["temp_count"] = False
    conf_text["count_annot"] = count_drv_text
    conf_visual["count_annot"] = count_drv_visual
    count_drv_all = [count_drv_text[ind] + count_drv_visual[ind] for ind in range(conf_all.shape[0])]
    conf_all["count_annot"] = count_drv_all
    # Get the count & type of confusions for each DRV
    count_text_confs, count_visual_confs, count_all_confs = [], [], []
    for drv in all_values:     
        # text confusions
        df_sub_t = df[df["variable"] == f'rel-value_text_conf_{drv}']
        found_values, found_dict = list(df_sub_t["value"]), {}
        for item in found_values:
            for value in clean_list_like(item):
                if value not in found_dict.keys():
                    found_dict[value] = 1
                else:
                    found_dict[value] += 1
        conf_text_count = [found_dict[key] if key in found_dict.keys() else 0 for key in all_values]
        count_text_confs.append(conf_text_count)
        # visual confusions
        df_sub_v = df[df["variable"] == f'rel-value_visual_conf_{drv}']
        found_values, found_dict = list(df_sub_v["value"]), {}
        for item in found_values:
            for value in clean_list_like(item):
                if value not in found_dict.keys():
                    found_dict[value] = 1
                else:
                    found_dict[value] += 1
        conf_visual_count = [found_dict[key] if key in found_dict.keys() else 0 for key in all_values]           
        count_visual_confs.append(conf_visual_count)
        # all confusions
        count_all_confs.append(np.add(conf_text_count, conf_visual_count))
    # transpose for insertion in the dataframe as new vectors
    count_text_confs_tr = np.transpose(count_text_confs)
    count_visual_confs_tr = np.transpose(count_visual_confs)
    count_all_confs_tr = np.transpose(count_all_confs)
    for pos in range(len(all_values)):
        for drv in all_values:
            conf_text[drv] = count_text_confs_tr[pos]
            conf_visual[drv] = count_visual_confs_tr[pos]
            conf_all[drv] = count_all_confs_tr[pos]
    # Get the count of annotations for each value with a confusion attached
    conf_text["count_dirty"] = conf_text.apply(lambda x: sum([x[drv] for drv in all_values]), axis=1)
    conf_visual["count_dirty"] = conf_visual.apply(lambda x: sum([x[drv] for drv in all_values]), axis=1)
    conf_all["count_dirty"] = conf_all.apply(lambda x: sum([x[drv] for drv in all_values]), axis=1)
    # Get the count of annotations with no confusion attached
    conf_text["count_clean"] = conf_text.apply(lambda x: x.count_annot - x.count_dirty, axis=1)
    conf_visual["count_clean"] = conf_visual.apply(lambda x: x.count_annot - x.count_dirty, axis=1)
    conf_all["count_clean"] = conf_all.apply(lambda x: x.count_annot - x.count_dirty, axis=1)
    # Get the count of times a DRV has been flagged as confusion
    total_conf_text, total_conf_visual, total_conf_all = [], [], []
    for drv in all_values:
        total_conf_text.append(conf_text[drv].sum())
        total_conf_visual.append(conf_visual[drv].sum())
        total_conf_all.append(conf_all[drv].sum())
    conf_text["count_flagged"] = total_conf_text
    conf_visual["count_flagged"] = total_conf_visual
    conf_all["count_flagged"] = total_conf_all
    # Get the count of times a DRV has been involved in a confusion
    conf_text["count_conf"] = conf_text.apply(lambda x: x["count_dirty"] + x["count_flagged"], axis=1)
    conf_visual["count_conf"] = conf_visual.apply(lambda x: x["count_dirty"] + x["count_flagged"], axis=1)
    conf_all["count_conf"] = conf_all.apply(lambda x: x["count_dirty"] + x["count_flagged"], axis=1)
    # Get the proportion of confused/total annotations and reorder features
    conf_text["conf/total"] = conf_text.apply(lambda x: get_proportion(x.count_clean, x.count_conf), axis=1)
    conf_visual["conf/total"] = conf_visual.apply(lambda x: get_proportion(x.count_clean, x.count_conf), axis=1)
    conf_all["conf/total"] = conf_all.apply(lambda x: get_proportion(x.count_clean, x.count_conf), axis=1)
    order = ["drv", "count_clean", "count_conf", "conf/total", *all_values]
    conf_text = conf_text[order]
    conf_visual = conf_visual[order]
    conf_all = conf_all[order]
    # Get the sum of the annotations-confusions matrix
    matrix_text = list(conf_text.apply(lambda x: [x[value] for value in all_values], axis=1))
    matrix_text_new = transform_matrix(matrix_text)
    matrix_visual = list(conf_visual.apply(lambda x: [x[value] for value in all_values], axis=1))
    matrix_visual_new = transform_matrix(matrix_visual)
    matrix_all = list(conf_all.apply(lambda x: [x[value] for value in all_values], axis=1))
    matrix_all_new = transform_matrix(matrix_all)
    for pos, drv in enumerate(all_values):
        conf_text[drv] = matrix_text_new[pos]
        conf_visual[drv] = matrix_visual_new[pos]
        conf_all[drv] = matrix_all_new[pos]
    # Save results as Excel file
    #### TODO: CURRENTLY DUMPED TO THE MAIN DIR, SHOULD IMPROVE THIS ####
    # po.path_for(SETTINGS.annotations_dir,alternative=f"confusion_{po.id}", ext="xlsx")
    with pd.ExcelWriter("confusions.xlsx") as writer:
        conf_all.to_excel(writer, sheet_name="all", index=False)
        conf_text.to_excel(writer, sheet_name="text", index=False)
        conf_visual.to_excel(writer, sheet_name="visual", index=False)

    return

    # standard shape
    df = (
        df[df["variable"].str.startswith("rel-value")]
        .drop(["task_id", "ann_id", "ts", "type"], axis=1)
        .reset_index(drop=True)
    )
    # exploded shape
    df = df.set_index(["platform_id", "user_id", "variable", "idx"])
    # df['rel-val-index'] = range(len(df))
    df = df.explode("value")

    df["value-index"] = df.groupby(
        level=["platform_id", "user_id", "variable", "idx"]
    ).cumcount()
    # sorted by re-value, and for each the confusion
    df = df.reset_index()
    dest = po.path_for(
        SETTINGS.temp_file_path,
        alternative="rel-values_confusions",
        ext=".csv",
    )
    df.to_csv(dest)
    print(dest)
    return dest
    # todo: separate conf out
    rel_values = [
        "personal-identity",
        "cultural-identity",
        "social-responsibility",
        "social-cohesion",
        "social-memory",
        "social-relations",
        "sense-of-place",
        "sense-of-agency",
        "spirituality",
        "stewardship-principle",
        "stewardship-eudaimonia",
        "literacy",
        "livelihoods",
        "well-being",
        "aesthetics",
        "reciprocity",
        "good-life",
        "kinship",
    ]
    df["value_cat"] = pd.Categorical(
        df["value"], categories=rel_values, ordered=True
    )
    df = df.sort_values("value_cat").drop(columns=["value_cat"])
    pass
    # next, for each value, check, if there is a matching conf:
    # p_id,u_id, idx must be matching
    dest = po.path_for(
        SETTINGS.temp_file_path,
        alternative="rel-values_confusions",
        ext=".csv",
    )
    # df.to_csv(dest)
