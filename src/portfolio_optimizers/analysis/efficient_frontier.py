import gurobipy as gp
import pandas as pd
import numpy as np
from itertools import product

from portfolio_optimizers.optimizers.multi_period_optimizer import MultiPeriodOptimizer
from portfolio_optimizers.optimizers.multi_period_optimizer import PortfolioConfiguration
from portfolio_optimizers.signals.market_signals import GARCHSignals, SyntheticSignals

class EfficientFrontierRunner:

    def calculate_efficient_frontier(self, 
                                     param_sweeps: dict, 
                                     market_data: pd.DataFrame,
                                     signal_type: str = "garch") -> pd.DataFrame:
        """ 
        Calculate the efficient frontier for a range of risk aversion levels and time horizons. 
        """
        param_combinations = self.parameter_sweep(param_sweeps)
        result_container = []
        for combo in param_combinations:
            config = {
                **self.build_rebalance_problem(market_data, time_horizon=combo["time_horizon"], signal_type=signal_type),
                "risk_aversion": combo["risk_aversion"],
                "time_horizon": combo["time_horizon"]
            }
            
            portfolio_config = PortfolioConfiguration(config)
            with gp.Env(empty=True) as env:
                env.start()

                with MultiPeriodOptimizer(env) as optimizer:
                    print("\nSetting up optimizer data...")
                    optimizer.set_data(portfolio_config)

                    print("\nBuilding optimization model...")
                    optimizer.build_model()

                    print("\nSolving optimization problem...")
                    status = optimizer.solve()

                    if status == "OPTIMAL":
                        solution = optimizer.get_solution()
                        solution_df = solution['solution'].copy()
                        solution_df["Objective_Value"] = solution["objective_value"]
                        solution_df["Max_Time_Horizon"] = combo["time_horizon"]
                        result_container.append(solution_df)

        return pd.concat(result_container)

    def parameter_sweep(self, param_sweeps: dict) -> list:
        """
        Generate all combinations of parameters for the efficient frontier calculation.
        """
        keys, values = zip(*param_sweeps.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        return param_combinations

    def build_rebalance_problem(self,
                                market_data: pd.DataFrame, 
                                time_horizon: int = 2,
                                signal_type: str = "garch") -> dict:
        """
        Build the rebalance problem dictionary with market data and forecasted returns and volatilities.
        """
        rebalance_problem = {
            "current_weights": None,
            "risk_aversion": 0,
            "market_data": market_data,
            "apply_shrinkage": True,
            "time_horizon": time_horizon,
            "buy_costs": [0.002 + (0.005 * t) for t in range(time_horizon)],
            "sell_costs": [0.002 + (0.005 * t) for t in range(time_horizon)],
            # "buy_costs": np.array([0.002] * time_horizon, dtype=float),
            # "sell_costs": np.array([0.001] * time_horizon, dtype=float),
            "hold_cost": 0.0005,
            "period_turnover_limit": 0.10,
            "global_horizon_turnover_limit": 0.3
        }

        signal = self._get_market_signal(market_data, signal_type=signal_type)
        mu_levels, sigma_levels = signal.forecast(horizon=rebalance_problem["time_horizon"])

        return {
            **rebalance_problem,
            "mu_levels": mu_levels,
            "sigma_levels": sigma_levels,
        }

    def _get_market_signal(self, market_data: pd.DataFrame, signal_type: str = "garch") -> object:
        """
        Get the market signal object based on the specified type.
        """
        if signal_type == "garch":
            return GARCHSignals(market_data, window=20)
        elif signal_type == "synthetic":
            return SyntheticSignals(market_data, window=20)
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")