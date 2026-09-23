import gurobipy as gp
import pandas as pd
import numpy as np

from portfolio_optimizers.optimizers.multi_period_optimizer import MultiPeriodOptimizer
from portfolio_optimizers.optimizers.multi_period_optimizer import PortfolioConfiguration

class EfficientFrontierRunner:
    def calculate_efficient_frontier(self, 
                                     risk_aversion_levels: np.ndarray, 
                                     rebalance_problem: dict) -> pd.DataFrame:
        result_container = []
        for risk_aversion in risk_aversion_levels:
            config = {
                **rebalance_problem,
                "risk_aversion": risk_aversion
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
                        result_container.append(solution_df)

        return pd.concat(result_container)