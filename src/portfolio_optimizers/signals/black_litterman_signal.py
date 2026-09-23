import numpy as np
import pandas as pd

class BlackLittermanSignal:
    def __init__(self, 
                 current_weights: np.ndarray,
                 sigma: pd.DataFrame,
                 prices: pd.DataFrame):

        self.mean_reversion_window = 4
        self.tau = 0.05
        self.delta = 2.5
        self.view_direction = "momentum"
        self.reversion_view = 0.03
        self.current_weights = current_weights
        self.sigma = sigma
        self.prices = prices

    def mean_returns(self) -> np.ndarray:
        """
        Returns the Black-Litterman posterior mean return vector. If no
        black_litterman config is present, falls back to the parent class
        historical mean returns.
        """
        pi = self._compute_equilibrium_returns(self.sigma)
        P, Q, omega = self._build_views(self.sigma)
        if not np.any(P):
            return pi  # no valid view (too few assets); fall back to equilibrium returns
        return self._compute_posterior(pi, self.sigma, P, Q, omega)
    
    def _compute_equilibrium_returns(self, sigma: pd.DataFrame) -> np.ndarray:
        """
        Computes the CAPM-implied equilibrium excess returns (pi) using reverse
        optimization: pi = delta * Sigma * w, where delta is the risk aversion
        coefficient and w is the current portfolio weight vector.
        """
        return self.delta * sigma @ self.current_weights

    def _build_views(self, sigma: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the investor view matrices (P, Q, Omega) using a mean-reversion
        signal. Assets in the bottom quintile by recent returns are expected to
        outperform assets in the top quintile (losers beat winners). Returns:
          P     — (1 x N) pick matrix encoding the relative view.
          Q     — (1,) array of the expected return spread.
          Omega — (1 x 1) diagonal uncertainty matrix scaled by tau * P @ Sigma @ P'.
        """        
        ranked = self._get_ranked_scores()

        n = len(ranked)
        quintile = n // 5

        losers  = ranked <= quintile
        winners = ranked > n - quintile

        P = self._determine_view_direction(n, winners, losers)
        Q = np.array([self.reversion_view])
        omega = np.diag(np.diag(self.tau * P @ sigma @ P.T))
        return P, Q, omega

    def _compute_posterior(self, 
                           pi: np.ndarray, 
                           sigma: pd.DataFrame, 
                           P: np.ndarray, 
                           Q: np.ndarray, 
                           omega: np.ndarray) -> np.ndarray:
        """
        Combines the equilibrium returns (pi) with the investor views (P, Q, Omega)
        using the Black-Litterman formula to produce a blended posterior mean vector:
          mu_BL = M @ (inv(tau*Sigma) @ pi + P' @ inv(Omega) @ Q)
        where M = inv(inv(tau*Sigma) + P' @ inv(Omega) @ P).
        """
        tau_sigma = self.tau * sigma
        inverted_tau_sigma = np.linalg.inv(tau_sigma)
        inverted_omega = np.linalg.inv(omega)

        M = np.linalg.inv(
            inverted_tau_sigma + P.T @ inverted_omega @ P
        )

        info_vector = inverted_tau_sigma @ pi + P.T @ inverted_omega @ Q
        return M @ info_vector
    
    def _get_ranked_scores(self):
        """
        Returns (ranked, expected_spread) used to construct the view matrix P.
        """
        short_returns = self.prices.pct_change(self.mean_reversion_window).iloc[-1]
        return short_returns.rank()
    
    def _determine_view_direction(self, 
                                  n: int, 
                                  winners: np.ndarray, 
                                  losers: np.ndarray) -> np.ndarray:
        """
        Builds the (1 x N) pick matrix P encoding a single long/short relative view based on the configured view_direction:
          "momentum"       — long winners, short losers (trend-following).
          "mean_reversion" — long losers, short winners (contrarian).
        Each leg is equally weighted and normalised so the row sums to zero.
        Defaults to mean_reversion if view_direction is unrecognised.
        """
        P = np.zeros((1, n))

        if winners.sum() == 0 or losers.sum() == 0:
            return P  # too few assets to form a valid view; express no view

        if self.view_direction == "momentum":
            P[0, winners] =  1 / winners.sum()  # long winners equally
            P[0, losers]  = -1 / losers.sum()   # short losers equally
        else:  # "mean_reversion" or unrecognised
            P[0, losers]  =  1 / losers.sum()   # long losers equally
            P[0, winners] = -1 / winners.sum()  # short winners equally

        return P