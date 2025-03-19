import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns


def lights_kappa(ratings, categories=None):
    """
    Calculate Light's Kappa for multiple raters.

    Parameters:
    -----------
    ratings : numpy.ndarray or pandas.DataFrame
        A 2D array where rows represent items and columns represent raters.
        Each cell contains the rating assigned by that rater to that item.
    categories : list, optional
        A list of all possible categories that can be assigned. If None,
        categories are inferred from the data.

    Returns:
    --------
    float
        Light's Kappa value (average of all pairwise Cohen's Kappa values)
    dict
        Dictionary of all pairwise Cohen's Kappa values where keys are tuples of rater indices
    pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    """
    if isinstance(ratings, pd.DataFrame):
        ratings = ratings.values

    # Ensure ratings is a 2D array
    if ratings.ndim != 2:
        raise ValueError("Ratings must be a 2D array where rows are items and columns are raters")

    num_items, num_raters = ratings.shape

    if num_raters < 2:
        raise ValueError("At least two raters are required")

    # Calculate Cohen's Kappa for all pairs of raters
    pairwise_kappas = {}
    for i, j in combinations(range(num_raters), 2):
        # Skip pairs where either rater has missing values
        mask = ~(np.isnan(ratings[:, i]) | np.isnan(ratings[:, j]))
        if np.sum(mask) == 0:
            pairwise_kappas[(i, j)] = np.nan
            continue

        if categories:
            # Use specified categories for calculation
            kappa = cohen_kappa_score(
                ratings[mask, i],
                ratings[mask, j],
                labels=categories
            )
        else:
            kappa = cohen_kappa_score(
                ratings[mask, i],
                ratings[mask, j]
            )
        pairwise_kappas[(i, j)] = kappa

    # Create a matrix of pairwise kappas
    kappa_matrix = np.zeros((num_raters, num_raters))
    np.fill_diagonal(kappa_matrix, 1.0)  # Perfect agreement with self

    for (i, j), kappa in pairwise_kappas.items():
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa  # Symmetric

    # Calculate Light's Kappa (average of pairwise kappas)
    valid_kappas = [k for k in pairwise_kappas.values() if not np.isnan(k)]
    if not valid_kappas:
        return np.nan, pairwise_kappas, pd.DataFrame(kappa_matrix)

    lights_k = np.mean(valid_kappas)

    return lights_k, pairwise_kappas, pd.DataFrame(kappa_matrix)


def plot_kappa_heatmap(kappa_matrix, rater_names=None):
    """
    Plot a heatmap of pairwise Cohen's Kappa values.

    Parameters:
    -----------
    kappa_matrix : pandas.DataFrame
        Matrix of pairwise Cohen's Kappa values
    rater_names : list, optional
        List of rater names to use for axis labels

    Returns:
    --------
    matplotlib.figure.Figure
    """
    if rater_names is not None:
        kappa_matrix.index = rater_names
        kappa_matrix.columns = rater_names

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(kappa_matrix, dtype=bool), k=1)

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
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title("Pairwise Cohen's Kappa Values")
    plt.tight_layout()
    return plt.gcf()


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


def analyze_rater_pairs(ratings, rater_names=None, categories=None):
    """
    Perform comprehensive analysis of inter-rater reliability using Light's Kappa.

    Parameters:
    -----------
    ratings : numpy.ndarray or pandas.DataFrame
        A 2D array where rows represent items and columns represent raters
    rater_names : list, optional
        List of names for each rater
    categories : list, optional
        List of all possible rating categories

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    if isinstance(ratings, pd.DataFrame) and rater_names is None:
        rater_names = ratings.columns.tolist()

    if rater_names is None:
        rater_names = [f"Rater {i + 1}" for i in range(ratings.shape[1])]

    # Calculate Light's Kappa
    lights_k, pairwise_kappas, kappa_matrix = lights_kappa(ratings, categories)

    # Replace index and columns with rater names
    kappa_matrix.index = rater_names
    kappa_matrix.columns = rater_names

    # Create a more readable version of pairwise kappas
    named_pairwise = {}
    for (i, j), kappa in pairwise_kappas.items():
        named_pairwise[f"{rater_names[i]} - {rater_names[j]}"] = kappa

    # Sort pairs by agreement level
    sorted_pairs = sorted(named_pairwise.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -2)

    # Find the pair with highest and lowest agreement
    valid_pairs = [(name, k) for name, k in named_pairwise.items() if not np.isnan(k)]
    best_pair = max(valid_pairs, key=lambda x: x[1]) if valid_pairs else (None, None)
    worst_pair = min(valid_pairs, key=lambda x: x[1]) if valid_pairs else (None, None)

    # Create summary
    results = {
        "lights_kappa": lights_k,
        "interpretation": interpret_kappa(lights_k),
        "pairwise_kappas": named_pairwise,
        "kappa_matrix": kappa_matrix,
        "best_pair": best_pair,
        "worst_pair": worst_pair,
        "sorted_pairs": sorted_pairs
    }

    return results


# Example usage
if __name__ == "__main__":
    # Example data: 10 items rated by 4 raters on a scale from 1-5
    np.random.seed(42)

    # Create some data with varying levels of agreement
    base_ratings = np.random.randint(1, 6, size=(20, 1))

    # Rater 1 and 2 tend to agree
    noise_level = 0.3
    rater1 = base_ratings + np.random.normal(0, noise_level, size=(20, 1)).astype(int)
    rater2 = base_ratings + np.random.normal(0, noise_level, size=(20, 1)).astype(int)

    # Rater 3 has moderate agreement
    rater3 = base_ratings + np.random.normal(0, 1, size=(20, 1)).astype(int)

    # Rater 4 has low agreement
    rater4 = np.random.randint(1, 6, size=(20, 1))

    # Combine and clip to valid range
    all_ratings = np.hstack([rater1, rater2, rater3, rater4])
    all_ratings = np.clip(all_ratings, 1, 5)

    # Convert to DataFrame
    df_ratings = pd.DataFrame(
        all_ratings,
        columns=["Expert A", "Expert B", "Expert C", "Novice"]
    )

    # Analyze the data
    results = analyze_rater_pairs(df_ratings, categories=[1, 2, 3, 4, 5])

    # Print results
    print(f"Light's Kappa: {results['lights_kappa']:.3f}")
    print(f"Interpretation: {results['interpretation']}")
    print("\nPairwise Cohen's Kappa values:")
    for pair, kappa in results['sorted_pairs']:
        print(f"  {pair}: {kappa:.3f} - {interpret_kappa(kappa)}")

    # Plot heatmap
    plot_kappa_heatmap(results['kappa_matrix'])
    plt.show()