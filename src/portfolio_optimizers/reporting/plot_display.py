import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np

class EfficientFrontierPlotter:

    def plot_efficient_frontier(self, 
                                results: pd.DataFrame,
                                risk_aversion_levels: np.ndarray) -> None:
        returns = np.zeros(risk_aversion_levels.shape)
        npos = np.zeros(risk_aversion_levels.shape)
        for i, risk_aversion in enumerate(risk_aversion_levels):
            result = results.loc[
                (results["Risk_Aversion"] == risk_aversion)
                & (results["Period"] == 1)
            ]
            if result.empty:
                continue

            returns[i] = result["Period_Return"].iloc[0]
            npos[i] = result["Active_Positions"].iloc[0]

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))

        # Axis 0: The efficient frontier
        axs[0].scatter(x=risk_aversion_levels, y=returns, marker="o", label="sample points", color="Red")
        axs[0].plot(risk_aversion_levels, returns, label="efficient frontier", color="Red")
        axs[0].set_xlabel("Risk Aversion")
        axs[0].set_ylabel("Expected return")
        axs[0].legend()
        axs[0].grid()

        # Axis 1: The number of open positions
        axs[1].scatter(x=risk_aversion_levels, y=npos, color="Red")
        axs[1].plot(risk_aversion_levels, npos, color="Red")
        axs[1].set_xlabel("Risk Aversion")
        axs[1].set_ylabel("Number of positions")
        axs[1].grid()

        plt.show()

    def plot_portfolio_composition(self, results: pd.DataFrame) -> None:
        results = results[results["Period"] == 1]

        composition = (
            results
            .reset_index()
            .pivot_table(
                index="Risk_Aversion",
                columns="Security",
                values="Weight",
                aggfunc="first",
            )
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sort_index()
            .astype(float)
        )

        my_cmap = LinearSegmentedColormap.from_list(
            "non-extreme gray",
            ["#111111", "#eeeeee"],
            N=256,
            gamma=1.0,
        )

        composition.plot.area(
            colormap=my_cmap,
            xlabel="Risk Aversion",
            ylabel="Portfolio Weight",
            figsize=(10, 4),
            linewidth=0,
        )

        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
        plt.tight_layout()
        plt.show()