from typing import Literal

import numpy as np

from ls_helper.models import ResultStruct, TaskAnnotResults


def fleiss_kappa(table, method='fleiss'):
    """Fleiss' and Randolph's kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns. Convert raw data
        into this format by using
        :func:`statsmodels.stats.inter_rater.aggregate_raters`
    method : str
        Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.

    Returns
    -------
    kappa : float
        Fleiss's or Randolph's kappa statistic for inter rater agreement

    Notes
    -----
    no variance or hypothesis tests yet

    Interrater agreement measures like Fleiss's kappa measure agreement relative
    to chance agreement. Different authors have proposed ways of defining
    these chance agreements. Fleiss' is based on the marginal sample distribution
    of categories, while Randolph uses a uniform distribution of categories as
    benchmark. Warrens (2010) showed that Randolph's kappa is always larger or
    equal to Fleiss' kappa. Under some commonly observed condition, Fleiss' and
    Randolph's kappa provide lower and upper bounds for two similar kappa_like
    measures by Light (1971) and Hubert (1977).

    References
    ----------
    Wikipedia https://en.wikipedia.org/wiki/Fleiss%27_kappa

    Fleiss, Joseph L. 1971. "Measuring Nominal Scale Agreement among Many
    Raters." Psychological Bulletin 76 (5): 378-82.
    https://doi.org/10.1037/h0031619.

    Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater
    K [free]): An Alternative to Fleiss' Fixed-Marginal Multirater Kappa."
    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005
    https://eric.ed.gov/?id=ED490661

    Warrens, Matthijs J. 2010. "Inequalities between Multi-Rater Kappas."
    Advances in Data Analysis and Classification 4 (4): 271-86.
    https://doi.org/10.1007/s11634-010-0073-4.
    """

    table = 1.0 * np.asarray(table)   #avoid integer division
    n_sub, n_cat =  table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    if method == 'fleiss':
        p_mean_exp = (p_cat*p_cat).sum()
    elif method.startswith('rand') or method.startswith('unif'):
        p_mean_exp = 1 / n_cat
    else: # fleiss
        p_mean_exp = (p_cat * p_cat).sum()

    if p_mean == 1:  # Perfect agreement
        return 1.0
    else:
        return (p_mean - p_mean_exp) / (1 - p_mean_exp)


def calculate_fleiss_kappa(results: list[TaskResults],
                           result_struct: ResultStruct,
                           choice_name: str,
                           num_coders: int) -> float:
    """
    Fleiss’ and Randolph’s kappa multi-rater agreement measure
    https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.fleiss_kappa.html
    """

    categories = result_struct.choices[choice_name]

    # restructure the data to get it to the structure for fleiss-kappa
    # subjects in rows, and categories in columns
    matrix: list[list[int]] = []
    extended_categories = categories + [NO_RESPONSE]
    for row in results:
        row_data = [0] * len(extended_categories)
        field_data = row.annotations.choices[choice_name]

        for i, category in enumerate(extended_categories):
            #print(category)
            row_data[i] = len(field_data[category])

        matrix.append(row_data)

    # if CONFIG.DEBUG_MODE:
    #     debug.store_coding_matrix(choice_name, matrix)
    kappa = fleiss_kappa(matrix)
    return float(kappa)


def calc_agreements(results_list: list[TaskAnnotResults],
                    result_struct: ResultStruct,
                    all_coders: set[str],
                    handle_missing_coding: Literal["only_full", "all", "discard_missing"] = 'only_full'):
    """

    :param results:
    :param result_struct:
    :param all_coders:
    :param handle_missing_coding:
    :return:
    """
    if len(all_coders) < 2:
        print("No agreement calculation for less than 2 coders. bye")
        return
    agreements = {}
    for choice in result_struct.choices.keys():
        if handle_missing_coding == "only_full":
            # check, if any coder, did not answer
            complete_results_list = list(filter(lambda r: not r.annotations.choices[choice][NO_RESPONSE], results_list))
            if not complete_results_list:
                print(f"Choice: '{choice}' - List of rows annotated by all coders is empty.")
                agreements[choice] = 0
                continue
            print(f"Choice: '{choice}' - Number of complete rows: {len(complete_results_list)}")
            agreements[choice] = calculate_fleiss_kappa(complete_results_list, result_struct, choice, len(all_coders))
        elif handle_missing_coding == "all":
            agreements[choice] = calculate_fleiss_kappa(results_list, result_struct, choice, len(all_coders))
        elif handle_missing_coding == "discard_missing":
            # discard missing. take the rows, which have the highest numbers of coders
            count_groups: dict[int, list[dict]] = {}
            for row in results_list:
                row_ = deepcopy(row.annotations.choices[choice])
                del row_[NO_RESPONSE]
                # check if its empty
                coded_count = reduce(lambda a,b: a+b, (len(option_coders) for option_coders in row_.values()))
                if not coded_count == 0:
                    count_groups.setdefault(coded_count,[]).append(row_)
            highest_counts_rows = list(sorted(count_groups.items(), key=lambda x: len(x[1]), reverse=True))
            if highest_counts_rows[0][0] == 1:
                highest_counts_rows.pop(0)
            highest_count, rows_ = highest_counts_rows[0]
            print(f"Choice: '{choice}' - Highest count: {highest_count}, {len(rows_)} rows")
            agreements[choice] = calculate_fleiss_kappa2(rows_, result_struct, choice, len(all_coders))
    for k, v in agreements.items():
        agreements[k] = {"value": round(v, 3), "interpretation": interpretation_str(v)}
    return agreements


