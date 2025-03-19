import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score


def multilabel_cohens_kappa(rater1_sets, rater2_sets):
    """
    Calculate Cohen's Kappa for multi-label coding where each rater can assign
    multiple categories to each item.

    Parameters:
    -----------
    rater1_sets : list of sets
        List of sets, where each set contains the categories assigned by rater1 to an item
    rater2_sets : list of sets
        List of sets, where each set contains the categories assigned by rater2 to an item

    Returns:
    --------
    float
        Cohen's Kappa value for multi-label coding
    """
    if len(rater1_sets) != len(rater2_sets):
        raise ValueError("Both raters must have the same number of items")

    n_items = len(rater1_sets)

    # Calculate observed agreement
    agreement_sum = 0
    for set1, set2 in zip(rater1_sets, rater2_sets):
        # Jaccard similarity for sets: |intersection| / |union|
        if not set1 and not set2:  # Both sets are empty
            agreement_sum += 1
        elif not set1 or not set2:  # One set is empty
            agreement_sum += 0
        else:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            agreement_sum += intersection / union

    observed_agreement = agreement_sum / n_items

    # Calculate agreement expected by chance
    # For multi-label, we use the concept of label density
    all_categories = set()
    for s in rater1_sets + rater2_sets:
        all_categories.update(s)

    if not all_categories:  # No categories were assigned by any rater
        return 1.0  # Perfect agreement (both raters assigned nothing to all items)

    # Calculate label density for each category for each rater
    cat_density_r1 = {cat: sum(1 for s in rater1_sets if cat in s) / n_items for cat in all_categories}
    cat_density_r2 = {cat: sum(1 for s in rater2_sets if cat in s) / n_items for cat in all_categories}

    # Expected agreement based on label densities
    expected_agreement_sum = 0
    for cat in all_categories:
        # Probability of both raters including the category
        p_both_include = cat_density_r1[cat] * cat_density_r2[cat]
        # Probability of both raters excluding the category
        p_both_exclude = (1 - cat_density_r1[cat]) * (1 - cat_density_r2[cat])
        # Add to expected agreement
        expected_agreement_sum += p_both_include + p_both_exclude

    expected_agreement = expected_agreement_sum / len(all_categories)

    # Calculate Cohen's Kappa
    if expected_agreement == 1:
        return 1.0  # Perfect agreement when expected agreement is also perfect

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def multilabel_lights_kappa(ratings_sets, rater_names=None):
    """
    Calculate Light's Kappa for multi-label coding data.

    Parameters:
    -----------
    ratings_sets : list of list of sets or dict of dict of sets
        Either a list where each element corresponds to a rater, and contains a list of sets
        (one for each item), or a dictionary with rater names as keys and values as lists of sets.
    rater_names : list, optional
        List of rater names, required if ratings_sets is a list

    Returns:
    --------
    float
        Light's Kappa value (average of all pairwise Cohen's Kappa values)
    dict
        Dictionary of all pairwise Cohen's Kappa values
    pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    """
    # Convert to dictionary format if list is provided
    if isinstance(ratings_sets, list):
        if rater_names is None:
            rater_names = [f"Rater {i + 1}" for i in range(len(ratings_sets))]
        ratings_dict = {name: sets for name, sets in zip(rater_names, ratings_sets)}
    else:
        ratings_dict = ratings_sets
        rater_names = list(ratings_dict.keys())

    num_raters = len(ratings_dict)

    if num_raters < 2:
        raise ValueError("At least two raters are required")

    # Calculate Cohen's Kappa for all pairs of raters
    pairwise_kappas = {}
    for rater1, rater2 in combinations(rater_names, 2):
        # Extract ratings for both raters
        sets_rater1 = ratings_dict[rater1]
        sets_rater2 = ratings_dict[rater2]

        # Find overlapping items (both raters have rated)
        valid_items = []
        for i, (set1, set2) in enumerate(zip(sets_rater1, sets_rater2)):
            if set1 is not None and set2 is not None:  # None indicates missing rating
                valid_items.append(i)

        if not valid_items:
            pairwise_kappas[(rater1, rater2)] = np.nan
            continue

        # Extract valid sets
        valid_sets_r1 = [sets_rater1[i] for i in valid_items]
        valid_sets_r2 = [sets_rater2[i] for i in valid_items]

        # Calculate multi-label Cohen's Kappa
        kappa = multilabel_cohens_kappa(valid_sets_r1, valid_sets_r2)
        pairwise_kappas[(rater1, rater2)] = kappa

    # Create a matrix of pairwise kappas
    kappa_matrix = pd.DataFrame(
        np.eye(num_raters),  # Diagonal of 1s (perfect agreement with self)
        index=rater_names,
        columns=rater_names
    )

    for (rater1, rater2), kappa in pairwise_kappas.items():
        kappa_matrix.loc[rater1, rater2] = kappa
        kappa_matrix.loc[rater2, rater1] = kappa  # Symmetric

    # Calculate Light's Kappa (average of pairwise kappas)
    valid_kappas = [k for k in pairwise_kappas.values() if not np.isnan(k)]
    if not valid_kappas:
        return np.nan, pairwise_kappas, kappa_matrix

    lights_k = np.mean(valid_kappas)

    return lights_k, pairwise_kappas, kappa_matrix


def jaccard_based_lights_kappa(ratings_sets, rater_names=None):
    """
    An alternative implementation of Light's Kappa for multi-label coding,
    based directly on Jaccard similarity for agreement calculation.

    Parameters:
    -----------
    ratings_sets : list of list of sets or dict of dict of sets
        Either a list where each element corresponds to a rater, and contains a list of sets
        (one for each item), or a dictionary with rater names as keys and values as lists of sets.
    rater_names : list, optional
        List of rater names, required if ratings_sets is a list

    Returns:
    --------
    float
        Light's Kappa value (average of all pairwise Cohen's Kappa values)
    dict
        Dictionary of all pairwise Cohen's Kappa values
    pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    """
    # Similar implementation to multilabel_lights_kappa, but uses Jaccard-based calculation
    # This version is included as an alternative approach

    # Convert to dictionary format if list is provided
    if isinstance(ratings_sets, list):
        if rater_names is None:
            rater_names = [f"Rater {i + 1}" for i in range(len(ratings_sets))]
        ratings_dict = {name: sets for name, sets in zip(rater_names, ratings_sets)}
    else:
        ratings_dict = ratings_sets
        rater_names = list(ratings_dict.keys())

    num_raters = len(ratings_dict)

    if num_raters < 2:
        raise ValueError("At least two raters are required")

    # Calculate Jaccard-based agreement for all pairs of raters
    pairwise_agreements = {}
    for rater1, rater2 in combinations(rater_names, 2):
        # Extract ratings for both raters
        sets_rater1 = ratings_dict[rater1]
        sets_rater2 = ratings_dict[rater2]

        # Find overlapping items (both raters have rated)
        valid_jaccard_scores = []
        for set1, set2 in zip(sets_rater1, sets_rater2):
            if set1 is not None and set2 is not None:  # None indicates missing rating
                # Calculate Jaccard similarity
                if not set1 and not set2:  # Both empty
                    valid_jaccard_scores.append(1.0)
                elif not set1 or not set2:  # One empty
                    valid_jaccard_scores.append(0.0)
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    valid_jaccard_scores.append(intersection / union)

        if not valid_jaccard_scores:
            pairwise_agreements[(rater1, rater2)] = np.nan
            continue

        # Average Jaccard similarity
        avg_jaccard = np.mean(valid_jaccard_scores)
        pairwise_agreements[(rater1, rater2)] = avg_jaccard

    # Create a matrix of pairwise agreements
    agreement_matrix = pd.DataFrame(
        np.eye(num_raters),  # Diagonal of 1s (perfect agreement with self)
        index=rater_names,
        columns=rater_names
    )

    for (rater1, rater2), agreement in pairwise_agreements.items():
        agreement_matrix.loc[rater1, rater2] = agreement
        agreement_matrix.loc[rater2, rater1] = agreement  # Symmetric

    # Calculate Light's Kappa-like measure (average of pairwise Jaccard scores)
    valid_agreements = [a for a in pairwise_agreements.values() if not np.isnan(a)]
    if not valid_agreements:
        return np.nan, pairwise_agreements, agreement_matrix

    avg_agreement = np.mean(valid_agreements)

    return avg_agreement, pairwise_agreements, agreement_matrix


def one_hot_encode_sets(sets_list, all_categories=None):
    """
    Convert a list of sets to a one-hot encoded matrix.

    Parameters:
    -----------
    sets_list : list of sets
        List of sets, where each set contains the categories assigned to an item
    all_categories : set or list, optional
        Set of all possible categories. If None, inferred from the data.

    Returns:
    --------
    numpy.ndarray
        One-hot encoded matrix (items x categories)
    list
        List of category names
    """
    if all_categories is None:
        all_categories = set()
        for s in sets_list:
            if s is not None:
                all_categories.update(s)

    all_categories = sorted(all_categories)
    n_items = len(sets_list)
    n_cats = len(all_categories)

    # Create one-hot matrix
    one_hot = np.zeros((n_items, n_cats))
    for i, item_set in enumerate(sets_list):
        if item_set is not None:
            for cat in item_set:
                if cat in all_categories:
                    j = all_categories.index(cat)
                    one_hot[i, j] = 1

    return one_hot, all_categories


def multilabel_lights_kappa_scikit(ratings_sets, rater_names=None):
    """
    Calculate Light's Kappa for multi-label coding using scikit-learn's metrics.
    This approach converts sets to one-hot encoding and uses Jaccard similarity.

    Parameters:
    -----------
    ratings_sets : list of list of sets or dict of dict of sets
        Either a list where each element corresponds to a rater, and contains a list of sets
        (one for each item), or a dictionary with rater names as keys and values as lists of sets.
    rater_names : list, optional
        List of rater names, required if ratings_sets is a list

    Returns:
    --------
    float
        Light's Kappa value (average of all pairwise Cohen's Kappa values)
    dict
        Dictionary of all pairwise Cohen's Kappa values
    pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    """
    # Convert to dictionary format if list is provided
    if isinstance(ratings_sets, list):
        if rater_names is None:
            rater_names = [f"Rater {i + 1}" for i in range(len(ratings_sets))]
        ratings_dict = {name: sets for name, sets in zip(rater_names, ratings_sets)}
    else:
        ratings_dict = ratings_sets
        rater_names = list(ratings_dict.keys())

    num_raters = len(ratings_dict)

    if num_raters < 2:
        raise ValueError("At least two raters are required")

    # Gather all categories
    all_categories = set()
    for rater_sets in ratings_dict.values():
        for item_set in rater_sets:
            if item_set is not None:
                all_categories.update(item_set)

    # Calculate Jaccard similarity for all pairs of raters
    pairwise_jaccards = {}
    for rater1, rater2 in combinations(rater_names, 2):
        # Extract ratings for both raters
        sets_rater1 = ratings_dict[rater1]
        sets_rater2 = ratings_dict[rater2]

        # Find overlapping items (both raters have rated)
        valid_items = []
        for i, (set1, set2) in enumerate(zip(sets_rater1, sets_rater2)):
            if set1 is not None and set2 is not None:
                valid_items.append(i)

        if not valid_items:
            pairwise_jaccards[(rater1, rater2)] = np.nan
            continue

        # Extract valid sets
        valid_sets_r1 = [sets_rater1[i] for i in valid_items]
        valid_sets_r2 = [sets_rater2[i] for i in valid_items]

        # Convert to one-hot encoding
        one_hot_r1, _ = one_hot_encode_sets(valid_sets_r1, all_categories)
        one_hot_r2, _ = one_hot_encode_sets(valid_sets_r2, all_categories)

        # Calculate Jaccard similarity score using scikit-learn
        # Average over all items (axis=0) to get a single score
        jaccard = jaccard_score(one_hot_r1, one_hot_r2, average='samples')
        pairwise_jaccards[(rater1, rater2)] = jaccard

    # Create a matrix of pairwise Jaccard scores
    jaccard_matrix = pd.DataFrame(
        np.eye(num_raters),  # Diagonal of 1s (perfect agreement with self)
        index=rater_names,
        columns=rater_names
    )

    for (rater1, rater2), jaccard in pairwise_jaccards.items():
        jaccard_matrix.loc[rater1, rater2] = jaccard
        jaccard_matrix.loc[rater2, rater1] = jaccard  # Symmetric

    # Calculate Light's Kappa-like measure (average of pairwise Jaccard scores)
    valid_jaccards = [j for j in pairwise_jaccards.values() if not np.isnan(j)]
    if not valid_jaccards:
        return np.nan, pairwise_jaccards, jaccard_matrix

    avg_jaccard = np.mean(valid_jaccards)

    # Convert average Jaccard to Kappa-like score
    # Estimate chance agreement based on label density
    all_one_hot = []
    n_items = len(next(iter(ratings_dict.values())))

    for rater, sets in ratings_dict.items():
        one_hot, _ = one_hot_encode_sets(sets, all_categories)
        all_one_hot.append(one_hot)

    # Concatenate all one-hot matrices
    all_one_hot_concat = np.vstack(all_one_hot)

    # Calculate expected agreement by chance (based on label density)
    label_densities = all_one_hot_concat.mean(axis=0)
    chance_agreement = 0
    for density in label_densities:
        # Probability of both raters including the category
        p_both_include = density * density
        # Probability of both raters excluding the category
        p_both_exclude = (1 - density) * (1 - density)
        # Add to expected agreement
        chance_agreement += p_both_include + p_both_exclude

    chance_agreement /= len(label_densities)

    # Convert Jaccard to Kappa using the chance agreement
    if chance_agreement < 1.0:
        kappa = (avg_jaccard - chance_agreement) / (1 - chance_agreement)
    else:
        kappa = 1.0  # Perfect agreement when chance agreement is also perfect

    return kappa, pairwise_jaccards, jaccard_matrix


def interpret_agreement(value):
    """
    Interpret the strength of agreement based on Kappa value or Jaccard similarity.

    Parameters:
    -----------
    value : float
        The agreement value to interpret

    Returns:
    --------
    str
        Interpretation of the agreement value
    """
    if value < 0:
        return "Poor agreement (less than chance)"
    elif value < 0.2:
        return "Slight agreement"
    elif value < 0.4:
        return "Fair agreement"
    elif value < 0.6:
        return "Moderate agreement"
    elif value < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def plot_agreement_heatmap(agreement_matrix, title="Pairwise Agreement Values"):
    """
    Plot a heatmap of pairwise agreement values.

    Parameters:
    -----------
    agreement_matrix : pandas.DataFrame
        Matrix of pairwise agreement values
    title : str, optional
        Title for the heatmap

    Returns:
    --------
    matplotlib.figure.Figure
    """
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        agreement_matrix,
        annot=True,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


# Example usage
if __name__ == "__main__":
    # Example: Multi-label coding where coders can select multiple categories for each item

    # Define categories
    categories = ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]

    # Create example data: 15 items (e.g., movies) rated by 4 raters
    # Each rater assigns a set of genres (categories) to each movie
    np.random.seed(42)


    # Generate some random multi-label data with pattern:
    # - Expert A and B tend to agree (similar label sets)
    # - Expert C has moderate agreement
    # - Novice has low agreement

    # Function to create similar sets with controlled variation
    def create_similar_sets(base_set, similarity):
        """Create a similar set with controlled variation"""
        if not base_set:
            if np.random.random() < similarity:
                return set()
            else:
                n_labels = np.random.randint(1, 4)
                return set(np.random.choice(categories, n_labels, replace=False))

        new_set = set()

        # Keep each label with probability = similarity
        for label in base_set:
            if np.random.random() < similarity:
                new_set.add(label)

        # Add new labels with probability (1-similarity)/3
        for label in set(categories) - base_set:
            if np.random.random() < (1 - similarity) / 3:
                new_set.add(label)

        return new_set


    # Generate base label sets for each item
    n_items = 15
    base_sets = []
    for _ in range(n_items):
        n_labels = np.random.randint(0, 5)  # 0 to 4 labels per item
        if n_labels == 0:
            base_sets.append(set())
        else:
            base_sets.append(set(np.random.choice(categories, n_labels, replace=False)))

    # Generate sets for each rater with varying similarity
    expert_a_sets = [create_similar_sets(s, 0.9) for s in base_sets]  # High similarity to base
    expert_b_sets = [create_similar_sets(s, 0.85) for s in base_sets]  # High similarity to base
    expert_c_sets = [create_similar_sets(s, 0.6) for s in base_sets]  # Moderate similarity to base
    novice_sets = [create_similar_sets(s, 0.3) for s in base_sets]  # Low similarity to base

    # Introduce some missing ratings (None)
    for i in range(n_items):
        if np.random.random() < 0.2:  # 20% chance of missing rating
            if np.random.random() < 0.5:
                expert_c_sets[i] = None
            else:
                novice_sets[i] = None

    # Create a dictionary of ratings
    ratings_dict = {
        "Expert A": expert_a_sets,
        "Expert B": expert_b_sets,
        "Expert C": expert_c_sets,
        "Novice": novice_sets
    }

    # Print the first few items for inspection
    print("Multi-label ratings example (first 5 items):")
    for i in range(5):
        print(f"Item {i + 1}:")
        for rater, sets in ratings_dict.items():
            if sets[i] is not None:
                print(f"  {rater}: {sets[i]}")
            else:
                print(f"  {rater}: Not rated")
        print()

    # Calculate Light's Kappa for multi-label data
    print("\nUsing custom multi-label Cohen's Kappa:")
    lights_k, pairwise_kappas, kappa_matrix = multilabel_lights_kappa(ratings_dict)

    print(f"Light's Kappa: {lights_k:.3f}")
    print(f"Interpretation: {interpret_agreement(lights_k)}")

    print("\nPairwise agreement values:")
    for (rater1, rater2), kappa in sorted(pairwise_kappas.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -2,
                                          reverse=True):
        if not np.isnan(kappa):
            print(f"  {rater1} - {rater2}: {kappa:.3f} - {interpret_agreement(kappa)}")
        else:
            print(f"  {rater1} - {rater2}: NaN - Insufficient overlapping ratings")

    # Plot heatmap
    plot_agreement_heatmap(kappa_matrix, "Multi-label Light's Kappa")

    # Alternative: Use scikit-learn based approach
    print("\nUsing scikit-learn Jaccard-based approach:")
    sk_kappa, sk_pairwise, sk_matrix = multilabel_lights_kappa_scikit(ratings_dict)

    print(f"Light's Kappa (scikit): {sk_kappa:.3f}")
    print(f"Interpretation: {interpret_agreement(sk_kappa)}")

    # Alternative: Direct Jaccard similarity approach
    print("\nUsing direct Jaccard similarity:")
    jaccard_k, jaccard_pairwise, jaccard_matrix = jaccard_based_lights_kappa(ratings_dict)

    print(f"Average Jaccard similarity: {jaccard_k:.3f}")
    print(f"Interpretation: {interpret_agreement(jaccard_k)}")

    plt.show()