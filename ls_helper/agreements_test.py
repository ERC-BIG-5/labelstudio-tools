import pandas as pd


def inter_rater_fleiss_kappa():
    # Define the rating matrix
    ratings = pd.DataFrame([
        [10, 0, 0],  # Item 1
        [20, 2, 0],  # Item 2
        # [6, 1, 0],  # Item 3
        # [6, 1, 0],  # Item 4
        # [5, 1, 1],  # Item 5
    ])

    # Compute Fleiss' Kappa with varying raters
    # kappa = irr.fleiss_kappa(ratings, method='fleiss')
    #
    # print(f"Fleiss' Kappa: {kappa:.4f}")

    from irrCAC.raw import CAC
    cac_4raters = CAC(ratings)
    gwet = cac_4raters.gwet()
    print(gwet)


if __name__ == "__main__":
    inter_rater_fleiss_kappa()
