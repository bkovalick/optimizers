import pandas as pd
import numpy as np

from optimizers.multi_period_optimizer import EfficientFrontier, build_rebalance_problem
from optimizers.mean_variance_optimization import MeanVarianceOptimizer
from optimizers.tax_lot_optimizer import TaxLotOptimizer, RebalanceProblemBuilder
from optimizers.portfolio_optimizer import PortfolioOptimizer

def run_multi_period_optimization():
    market_data_df = pd.read_pickle("data/subset_weekly_closings_10yrs.pkl")
    rebalance_problem = build_rebalance_problem(market_data_df, time_horizon=2)

    risk_aversion_levels = np.linspace(0.25, 10, 25)
    frontier = EfficientFrontier()
    frontier.calculate_efficient_frontier(risk_aversion_levels, rebalance_problem)
    frontier.plot_efficient_frontier(risk_aversion_levels)
    frontier.plot_portfolio_composition()

def run_tax_lot_optimization():

    rebal_config = {
        "daily_price_file": "optimizers/daily_prices.csv",
        "tickers": ["AAPL", "MSFT", "XOM", "BAC", "UNH"],
        "native_tickers": ["AAPL", "MSFT", "XOM", "BAC", "UNH"],
        "portfolio_parameters": {
                "2018-02-06": { "amount": 100000, "weights": {"AAPL": 0.20, "MSFT": 0.20, "XOM": 0.20, "BAC": 0.20, "UNH": 0.20}
            }
        },
        "rebalance_date": "2024-04-08",
        "tax_preference": 1.0,
        "total_gains_budget": {"apply_budget": False, "amount": 100000 },
        "short_term_gains_budget": {"apply_budget": False, "amount": 100000 },
    }

    optimizer = TaxLotOptimizer()
    rebalance_problem = RebalanceProblemBuilder(rebal_config).build()
    tax_pref_results = []
    tax_preferences = np.arange(0.0, 1.01, 0.10)
    # tax_preferences = [0, 10000, 100000, 500000, 1e06, 3e06]
    for tax_prev in tax_preferences:
        rebalance_problem.config["tax_preference"] = tax_prev
        results = optimizer.optimize(rebalance_problem)
        results.insert(0, "TaxPreference", tax_prev)
        tax_pref_results.append(results)
    
    results = pd.concat(tax_pref_results, keys=tax_preferences, names=["TaxPreference", "Row"])
    results.to_clipboard(index=False)    

if __name__ == "__main__":
    # Example usage of the optimizers
    run_multi_period_optimization()
    # run_tax_lot_optimization()