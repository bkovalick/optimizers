import pandas as pd
import numpy as np
from pathlib import Path

from portfolio_optimizers.analysis.efficient_frontier import EfficientFrontierRunner
from portfolio_optimizers.reporting.plot_display import EfficientFrontierPlotter
from portfolio_optimizers.signals.market_signals import GARCHSignals, SyntheticSignals
from portfolio_optimizers.reporting.report_writer import OptimizerReportWriter
from portfolio_optimizers.analysis.parameter_sweep import RebalanceProblemSweep

SRC_DIR = Path(__file__).resolve().parents[1]

def get_market_signal(market_data: pd.DataFrame, 
                       signal_type: str = "garch") -> object:
    """
    Get the market signal object based on the specified type.
    """
    if signal_type == "garch":
        return GARCHSignals(market_data, window=20)
    elif signal_type == "synthetic":
        return SyntheticSignals(market_data, window=20)
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

def build_rebalance_problem(market_data: pd.DataFrame, 
                            time_horizon: int = 2,
                            signal_type: str = "garch") -> dict:
    """
    Build the rebalance problem dictionary with market data and forecasted returns and volatilities.
    """
    buy_costs = [0.002 + (0.0005 * t) for t in range(time_horizon)]
    sell_costs = [0.002 + (0.0005 * t) for t in range(time_horizon)]
    investment_universe = market_data.columns.tolist()
    investment_universe = ["AAPL", "MSFT", "XOM", "BAC", "UNH"]  # Example subset of tickers
    # investment_market_data = market_data[investment_universe]
    investment_market_data = market_data.copy()
    rebalance_problem = {
        "current_weights": None,
        "investment_universe": investment_universe,
        "risk_aversion": 0,
        "market_data": investment_market_data,
        "apply_shrinkage": True,
        "time_horizon": time_horizon,
        "buy_costs": buy_costs,
        "sell_costs": sell_costs,
        "hold_cost": 0.0005,
        "period_turnover_limit": 0.10,
        "global_horizon_turnover_limit": 0.3,
        "max_long": 1.3,
        "max_short": 0.30,
        "max_position": 0.10
    }

    signal = get_market_signal(investment_market_data, signal_type=signal_type)
    mu_levels, sigma_levels = signal.forecast(horizon=time_horizon)

    return {
        **rebalance_problem,
        "mu_levels": mu_levels,
        "sigma_levels": sigma_levels,
    }

def run_multi_period_optimization():
    """ Run the multi-period optimization using the Efficient Frontier approach. """
    market_data_df = pd.read_pickle(SRC_DIR / "data" / "subset_weekly_closings_10yrs.pkl")
    risk_aversion_levels = np.linspace(0.25, 4, 5)
    time_horizons = [2]
    # , 2, 5, 10] 
    param_sweeps = {"risk_aversion": risk_aversion_levels, "time_horizon": time_horizons}
    parameter_sweep = RebalanceProblemSweep(build_rebalance_problem)
    rebalance_problems = parameter_sweep.build(param_sweeps, market_data_df, signal_type="garch")
    
    frontier = EfficientFrontierRunner()
    frontier_results = frontier.calculate_efficient_frontier(rebalance_problems)

    writer = OptimizerReportWriter(SRC_DIR / "optimizer_results", 
                                   f"efficient_frontier_results_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx")
    writer.write_report(frontier_results)

    # plotter = EfficientFrontierPlotter()
    # plotter.plot_efficient_frontier(frontier_results, risk_aversion_levels)
    # plotter.plot_portfolio_composition(frontier_results)

def run_tax_lot_optimization():
    from portfolio_optimizers.optimizers.tax_lot_optimizer import TaxLotOptimizer, RebalanceProblemBuilder

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