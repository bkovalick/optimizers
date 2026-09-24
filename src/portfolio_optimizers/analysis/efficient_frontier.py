import gurobipy as gp
import pandas as pd

from portfolio_optimizers.optimizers.multi_period_optimizer import MultiPeriodOptimizer
from portfolio_optimizers.optimizers.multi_period_optimizer import PortfolioConfiguration

class EfficientFrontierRunner:

    def calculate_efficient_frontier(self, rebalance_problems: list) -> pd.DataFrame:
        """ 
        Calculate the efficient frontier for a range of risk aversion levels and time horizons. 
        """
        result_container = []
        for rebalance_problem in rebalance_problems:
            config = rebalance_problem.copy()
            
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
                        solution_df["Max_Time_Horizon"] = config["time_horizon"]
                        result_container.append(solution_df)

        return pd.concat(result_container)