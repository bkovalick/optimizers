import pandas as pd
import numpy as np
from pathlib import Path

from portfolio_optimizers.signals.market_signals import GARCHSignals, SyntheticSignals
from portfolio_optimizers.analysis.efficient_frontier import EfficientFrontierRunner
from portfolio_optimizers.reporting.plot_display import EfficientFrontierPlotter
from portfolio_optimizers.reporting.report_writer import OptimizerReportWriter
from portfolio_optimizers.optimizers.tax_lot_optimizer import TaxLotOptimizer, RebalanceProblemBuilder

SRC_DIR = Path(__file__).resolve().parents[1]

def build_rebalance_problem(market_data: pd.DataFrame, time_horizon: int = 2) -> dict:
    rebalance_problem = {
        "current_weights": None,
        "risk_aversion": 0,
        "market_data": market_data,
        "apply_shrinkage": True,
        "time_horizon": time_horizon,
        "buy_cost": 0.002,
        "sell_cost": 0.001,
        "hold_cost": 0.0005,
    }
    # synthetic_signals = SyntheticSignals(rebalance_problem["market_data"])
    # mu_levels, sigma_levels = synthetic_signals.forecast(horizon=rebalance_problem["time_horizon"])
    garch_signals = GARCHSignals(rebalance_problem["market_data"])
    mu_levels, sigma_levels = garch_signals.forecast(horizon=rebalance_problem["time_horizon"])

    return {
        **rebalance_problem,
        "mu_levels": mu_levels,
        "sigma_levels": sigma_levels,
    }

def run_multi_period_optimization():
    market_data_df = pd.read_pickle(SRC_DIR / "data" / "subset_weekly_closings_10yrs.pkl")
    rebalance_problem = build_rebalance_problem(market_data_df, time_horizon=1)

    risk_aversion_levels = np.linspace(0.25, 10, 25)
    frontier = EfficientFrontierRunner()
    frontier_results = frontier.calculate_efficient_frontier(risk_aversion_levels, rebalance_problem)

    time_horizon = rebalance_problem["time_horizon"]
    writer = OptimizerReportWriter(SRC_DIR / "optimizer_results", 
                                   f"efficient_frontier_results_{pd.Timestamp.now().strftime('%Y%m%d')}_time_H_{time_horizon}.xlsx")
    writer.write_report(frontier_results)

    plotter = EfficientFrontierPlotter()
    plotter.plot_efficient_frontier(frontier_results, risk_aversion_levels)
    plotter.plot_portfolio_composition(frontier_results)

def run_tax_lot_optimization():
    rebal_config = {
        "daily_price_file": SRC_DIR / "data" / "daily_prices.csv",
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
    print("Running Multi-Period Optimization\n...")
    run_multi_period_optimization()

    # print("Running Tax Lot Optimization...")
    # run_tax_lot_optimization()