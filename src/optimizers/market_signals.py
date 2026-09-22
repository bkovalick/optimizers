import pandas as pd
import numpy as np
import abc
from arch import arch_model 

class Signals(abc.ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = self.data.pct_change().iloc[1:] * 100

    @abc.abstractmethod
    def forecast(self, horizon: int = 1):
        pass

    def compute_ewma_returns(self, span: int = 20):
        # Compute exponentially weighted moving average (EWMA) returns
        ewma_returns = self.returns.ewm(span=span).mean()
        return ewma_returns

class GARCHSignals(Signals):
    def __init__(self, data: pd.DataFrame, window: int = 20):
        super().__init__(data)
        self.window = window

    def forecast(self, horizon: int = 1):
        """ Forecast future returns and volatility using GARCH(1, 1) model. """
        asset_variance_forecasts = []
        for security in self.returns.columns:
            asset_returns = self.returns[security].dropna().to_numpy(dtype=float)
            garch_model = arch_model(asset_returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_forecast = garch_fit.forecast(horizon=horizon)
            asset_variance_forecasts.append(garch_forecast.variance.values[-1])

        variance_forecasts = np.array(asset_variance_forecasts).T
        correlation_matrix = self.returns.corr().to_numpy(dtype=float)
        sigma_levels = []
        for t in range(horizon):
            vol_forecast = np.sqrt(variance_forecasts[t])
            sigma_levels.append(np.outer(vol_forecast, vol_forecast) * correlation_matrix)

        ewma_returns = self.compute_ewma_returns()
        mu_levels = [ewma_returns.iloc[-1].to_numpy(dtype=float).copy() for _ in range(horizon)]

        return mu_levels, sigma_levels

class SyntheticSignals(Signals):
    def __init__(self, data: pd.DataFrame, window: int = 20):
        super().__init__(data)
        self.window = window

    def forecast(self, horizon: int = 1, seed: int = 42):
        """ Generate synthetic signals based on historical returns and volatility. """
        rng = np.random.default_rng(seed)
        base_mu = self.returns.mean().to_numpy(dtype=float)
        base_sigma = self.returns.cov().to_numpy(dtype=float)

        mu_levels, sigma_levels = [], []
        for t in range(horizon):
            noise = rng.normal(0, 0.005, size=self.returns.shape[1])
            mu_t = base_mu + noise
            
            vol_scale = rng.uniform(0.85, 1.15)
            sigma_t = base_sigma * vol_scale ** 2

            mu_levels.append(mu_t)
            sigma_levels.append(sigma_t)

        return mu_levels, sigma_levels


def simulate_gbm(S0, alpha, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = np.random.standard_normal(size=N)
    W = np.zeros(N + 1)
    W[1:] = np.cumsum(np.random.standard_normal(size=N)) * np.sqrt(dt)
    X = S0 * np.exp((alpha - 0.5 * sigma**2) * t + sigma * W)
    return t, X

if __name__ == "__main__":
    # Example usage
    market_data_df = pd.read_pickle("data/subset_weekly_closings_10yrs.pkl")
    garch_signals = GARCHSignals(market_data_df, window=20)
    garch_signals = garch_signals.forecast()
    print("GARCH Signals:\n", garch_signals)

    synthetic_signals = SyntheticSignals(market_data_df, window=20).forecast()
    print("Synthetic Signals:\n", synthetic_signals)