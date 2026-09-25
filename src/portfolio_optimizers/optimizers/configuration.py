import pandas as pd
import numpy as np

class PortfolioConfiguration:
    def __init__(self, config: dict):
        self._config = config

    @property
    def securities(self) -> list:
        self._securities = self.market_data.columns
        if self._securities is None:
            raise ValueError("Securities are missing, likely due to missing market data.")
        return self._securities
        
    @property
    def n_constituents(self) -> int:
        self._n_constituents = len(self.securities) 
        if self._n_constituents is None:
            raise ValueError("Constituents are missing, likely due to missing market data.")

        return self._n_constituents  

    @property
    def market_data(self) -> pd.DataFrame: 
        self._market_data = self._config.get("market_data", None)
        if self._market_data is None:
            raise ValueError("No market data provided, therefore optimization is stopping")

        return self._market_data

    @property
    def market_returns(self) -> pd.DataFrame:
        self._market_returns = self.market_data.pct_change().iloc[1:] * 100
        if self._market_returns is None:
            raise ValueError("Market Returns are missing. Please provide proper market data")
        return self._market_returns
    
    @property
    def mu_levels(self) -> np.ndarray:
        self._mu_levels = self._config.get("mu_levels", None)
        if self._mu_levels is None:
            raise ValueError("Mean Returns are missing. Please provide proper market data")

        return self._mu_levels

    @property
    def sigma_levels(self) -> np.ndarray:
        self._sigma_levels = self._config.get("sigma_levels", None)
        if self._sigma_levels is None:
            raise ValueError("Sigma levels are missing. Please provide proper market data")                 

        return self._sigma_levels

    @property
    def current_weights(self) -> np.ndarray:
        current_weights = self._config.get("current_weights", None)
        if current_weights is None or len(current_weights) == 0:
            self._current_weights = np.repeat(1.0 / self.n_constituents, self.n_constituents)
        else:
            if len(current_weights) != self.n_constituents:
                raise ValueError("Current weights must match the number of market data columns")
            self._current_weights = np.asarray(current_weights, dtype=float)    

        return self._current_weights

    @property
    def apply_shrinkage(self) -> bool:
        return self._config.get("apply_shrinkage", False)

    @property
    def risk_aversion(self) -> float:
        return self._config.get("risk_aversion", 0.0)

    @property
    def variance_level(self) -> float:
        return self._config.get("variance_level", 3.5)

    @property
    def time_horizon(self) -> int:
        return self._config.get("time_horizon", 3)

    @property
    def buy_costs(self) -> float:
        return self._config.get("buy_costs", np.array([0.002] * self.n_constituents, dtype=float))

    @property
    def sell_costs(self) -> float:
        return self._config.get("sell_costs", np.array([0.001] * self.n_constituents, dtype=float))

    @property
    def hold_cost(self) -> float:
        return self._config.get("hold_cost", 0.0005)

    @property
    def period_turnover_limit(self) -> float:
        return self._config.get("period_turnover_limit", 0.10)
    
    @property
    def global_horizon_turnover_limit(self) -> float:
        return self._config.get("global_horizon_turnover_limit", 0.3)

    @property
    def terminal_weights(self) -> np.ndarray:
        terminal_weights = self._config.get("terminal_weights", None)
        if terminal_weights is None or len(terminal_weights) == 0:
            self._terminal_weights = np.repeat(1.0 / self.n_constituents, self.n_constituents)
        else:
            if len(terminal_weights) != self.n_constituents:
                raise ValueError("Terminal weights must match the number of market data columns")
            self._terminal_weights = np.asarray(terminal_weights, dtype=float)    

        return self._terminal_weights

    @property
    def net_target(self) -> float:
        return self.max_long - self.max_short

    @property
    def gross_exposure(self) -> float:
        return self.max_long + self.max_short

    @property
    def max_long(self) -> float:
        return self._config.get("max_long", 1.0)

    @property
    def max_short(self) -> float:
        return self._config.get("max_short", 0.0)

    @property
    def max_position(self) -> float:
        return self._config.get("max_position", 0.10)