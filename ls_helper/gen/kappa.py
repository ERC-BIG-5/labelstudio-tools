from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets (intersection / union).

    Parameters:
    -----------
    set1 : set
        First set of categories
    set2 : set
        Second set of categories

    Returns:
    --------
    float
        Jaccard similarity (0-1)
    """
    if not set1 and not set2:  # Both empty
        return 1.0
    elif not set1 or not set2:  # One empty
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def paired_multilabel_kappa(item_data):
    """
    Calculate Light's Kappa for paired multi-label data where each item is rated by exactly
    two coders who can assign multiple categories per item.

    Parameters:
    -----------
    item_data : list of tuples
        List of (coder1, coder2, set1, set2) tuples, where:
        - coder1, coder2 are the IDs or names of the two coders rating this item
        - set1, set2 are the sets of categories assigned by each coder

    Returns:
    --------
    dict
        Dictionary containing Light's Kappa results and additional analysis
    """
    # Step 1: Group the data by coder pairs
    pair_items = defaultdict(list)
    for coder1, coder2, set1, set2 in item_data:
        # Ensure consistent ordering of coder pairs
        if coder1 > coder2:
            coder1, coder2 = coder2, coder1
            set1, set2 = set2, set1

        pair_items[(coder1, coder2)].append((set1, set2))

    # Step 2: Calculate agreement for each coder pair
    pair_agreements = {}
    all_jaccard_scores = []

    for pair, item_sets in pair_items.items():
        jaccard_scores = []
        for set1, set2 in item_sets:
            jaccard = jaccard_similarity(set1, set2)
            jaccard_scores.append(jaccard)
            all_jaccard_scores.append((pair, jaccard))

        # Calculate average Jaccard similarity (observed agreement)
        observed_agreement = np.mean(jaccard_scores)

        # Calculate expected agreement based on category frequencies within this pair
        all_categories = set()
        cat_counts1 = defaultdict(int)
        cat_counts2 = defaultdict(int)

        for set1, set2 in item_sets:
            for cat in set1:
                all_categories.add(cat)
                cat_counts1[cat] += 1
            for cat in set2:
                all_categories.add(cat)
                cat_counts2[cat] += 1

        n_items = len(item_sets)

        if not all_categories:  # No categories were used
            # Perfect agreement if both coders assigned nothing to all items
            pair_agreements[pair] = 1.0
            continue

        # Calculate chance agreement based on category frequencies
        expected_agreement = 0
        for cat in all_categories:
            # Probability of both including the category
            p1 = cat_counts1[cat] / n_items
            p2 = cat_counts2[cat] / n_items
            p_both_include = p1 * p2

            # Probability of both excluding the category
            p_both_exclude = (1 - p1) * (1 - p2)

            expected_agreement += p_both_include + p_both_exclude

        expected_agreement /= len(all_categories)

        # Calculate Cohen's Kappa
        if expected_agreement < 1.0:
            kappa = (observed_agreement - expected_agreement) / (
                1 - expected_agreement
            )
        else:
            kappa = 1.0  # Perfect agreement when expected agreement is 1

        pair_agreements[pair] = kappa

    # Step 3: Calculate Light's Kappa (average of pair-wise kappas)
    lights_kappa = np.mean(list(pair_agreements.values()))

    # Step 4: Create the result dictionary with additional analysis
    unique_coders = set()
    for coder1, coder2 in pair_agreements.keys():
        unique_coders.add(coder1)
        unique_coders.add(coder2)

    # Create a matrix of pair-wise kappas
    n_coders = len(unique_coders)
    coder_indices = {coder: i for i, coder in enumerate(sorted(unique_coders))}

    kappa_matrix = np.eye(
        n_coders
    )  # Diagonal of 1s (perfect agreement with self)
    for (coder1, coder2), kappa in pair_agreements.items():
        i, j = coder_indices[coder1], coder_indices[coder2]
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa  # Symmetric

    # Create pandas DataFrame with coder names
    kappa_df = pd.DataFrame(
        kappa_matrix,
        index=sorted(unique_coders),
        columns=sorted(unique_coders),
    )

    # Identify items with highest and lowest agreement
    item_agreements = [
        (pair, i, jaccard)
        for i, (pair, jaccard) in enumerate(all_jaccard_scores)
    ]
    item_agreements.sort(key=lambda x: x[2])

    worst_agreements = (
        item_agreements[:5] if len(item_agreements) >= 5 else item_agreements
    )
    best_agreements = (
        item_agreements[-5:]
        if len(item_agreements) >= 5
        else reversed(item_agreements)
    )

    # Compile results
    results = {
        "lights_kappa": lights_kappa,
        "pair_agreements": pair_agreements,
        "kappa_matrix": kappa_df,
        "worst_agreements": worst_agreements,
        "best_agreements": list(reversed(best_agreements)),
        "interpretation": interpret_kappa(lights_kappa),
    }

    return results


def interpret_kappa(kappa_value):
    """
    Interpret the strength of agreement based on Kappa value.

    Parameters:
    -----------
    kappa_value : float
        The Kappa value to interpret

    Returns:
    --------
    str
        Interpretation of the Kappa value
    """
    if kappa_value < 0:
        return "Poor agreement (less than chance)"
    elif kappa_value < 0.2:
        return "Slight agreement"
    elif kappa_value < 0.4:
        return "Fair agreement"
    elif kappa_value < 0.6:
        return "Moderate agreement"
    elif kappa_value < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def plot_kappa_heatmap(kappa_matrix, title="Pairwise Cohen's Kappa Values"):
    """
    Plot a heatmap of pairwise Cohen's Kappa values.

    Parameters:
    -----------
    kappa_matrix : pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    title : str, optional
        Title for the heatmap

    Returns:
    --------
    matplotlib.figure.Figure
    """
    plt.figure(figsize=(10, 8))

    # Only mask the upper triangle if the matrix is square
    if kappa_matrix.shape[0] == kappa_matrix.shape[1]:
        mask = np.triu(np.ones_like(kappa_matrix, dtype=bool), k=1)
    else:
        mask = None

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        kappa_matrix,
        annot=True,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def create_test_data(n_items=30, n_coders=5, n_categories=8, sparsity=0.7):
    """
    Create test data for paired multi-label coding.

    Parameters:
    -----------
    n_items : int
        Number of items to create
    n_coders : int
        Number of coders
    n_categories : int
        Number of possible categories
    sparsity : float
        Proportion of item-coder pairs to include (1.0 = all pairs)

    Returns:
    --------
    list
        List of (coder1, coder2, set1, set2) tuples
    """
    np.random.seed(42)
    coders = [f"Coder_{i + 1}" for i in range(n_coders)]
    categories = [f"Cat_{chr(65 + i)}" for i in range(n_categories)]

    # Create base category sets for each item with varying complexity
    base_sets = []
    for _ in range(n_items):
        n_cats = np.random.randint(0, 5)  # 0 to 4 categories per item
        if n_cats == 0:
            base_sets.append(set())
        else:
            base_sets.append(
                set(np.random.choice(categories, n_cats, replace=False))
            )

    # Create coding pairs with varying agreement
    item_data = []

    # Each item is rated by approximately 2 coders
    for item_idx, base_set in enumerate(base_sets):
        # Randomly select 2 coders for this item
        selected_coders = np.random.choice(coders, 2, replace=False)

        # Create sets for the coders with controlled variation from base set
        coder_sets = []

        for coder in selected_coders:
            # Add noise based on coder's experience level (using coder number as proxy)
            coder_num = int(coder.split("_")[1])
            experience = (
                1 - (coder_num / n_coders) * 0.8
            )  # Higher number = less experienced

            # Create a similar set with controlled variation
            new_set = set()

            # Keep each label with probability based on experience
            for label in base_set:
                if np.random.random() < experience:
                    new_set.add(label)

            # Add new labels with probability based on inexperience
            for label in set(categories) - base_set:
                if np.random.random() < (1 - experience) / 3:
                    new_set.add(label)

            coder_sets.append(new_set)

        # Add to the data
        item_data.append(
            (
                selected_coders[0],
                selected_coders[1],
                coder_sets[0],
                coder_sets[1],
            )
        )

    return item_data


# Example usage
if __name__ == "__main__":
    # Create test data: Each item is rated by exactly 2 coders who can assign multiple categories
    item_data = create_test_data(n_items=40, n_coders=5, n_categories=8)

    # Print a few sample items
    print("Sample multi-label paired coding data:")
    for i, (coder1, coder2, set1, set2) in enumerate(item_data[:5]):
        print(f"Item {i + 1}:")
        print(f"  {coder1}: {set1}")
        print(f"  {coder2}: {set2}")
        print(f"  Jaccard similarity: {jaccard_similarity(set1, set2):.3f}")
        print()

    # Calculate Light's Kappa for the paired multi-label data
    results = paired_multilabel_kappa(item_data)

    # Print results
    print(
        f"Light's Kappa for paired multi-label coding: {results['lights_kappa']:.3f}"
    )
    print(f"Interpretation: {results['interpretation']}")

    print("\nPairwise Cohen's Kappa between coders:")
    for pair, kappa in sorted(
        results["pair_agreements"].items(), key=lambda x: x[1], reverse=True
    ):
        print(
            f"  {pair[0]} - {pair[1]}: {kappa:.3f} - {interpret_kappa(kappa)}"
        )

    # Plot heatmap of kappa values between coders
    plot_kappa_heatmap(results["kappa_matrix"])

    # Show examples of high and low agreement items
    print("\nItems with lowest agreement:")
    for pair, idx, jaccard in results["worst_agreements"]:
        print(f"  Item {idx}: {pair[0]}-{pair[1]} agreement = {jaccard:.3f}")

    print("\nItems with highest agreement:")
    for pair, idx, jaccard in results["best_agreements"]:
        print(f"  Item {idx}: {pair[0]}-{pair[1]} agreement = {jaccard:.3f}")

    plt.show()
