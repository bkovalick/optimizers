import numpy as np
import optimizers.mean_variance_optimization as opt
import optimizers.portfolio_optimizer as port
from optimizers.gain_loss_optimizer import GainLossOptimizer
from data.model_data import GurobiModelData, LiveMarketData
from mosek.fusion import *
from visualization.visualization_analysis import create_visualizations, print_detailed_analysis

import numpy as np
import gurobipy as gp
from itertools import product
import pandas as pd

def simple_portfolio_optimization(num_assets, expected_returns, cov_matrix, max_risk):
    with Model("markowitz_simple") as M:
        # Define the decision variables (asset weights x), which must be non-negative.
        x = M.variable('x', num_assets, Domain.greaterThan(0.0))

        # Add the budget constraint: the sum of weights must equal 1.
        M.constraint('budget', Expr.sum(x), Domain.equalsTo(1.0))

        # Add the risk constraint using a quadratic cone.
        # This constrains the portfolio's standard deviation (sqrt(x.T @ cov_matrix @ x)) to be less than or equal to max_risk.
        # It requires the Cholesky decomposition of the covariance matrix.
        G = np.linalg.cholesky(cov_matrix)
        M.constraint('risk', Expr.vstack(max_risk, Expr.mul(G.T, x)), Domain.inQCone())

        # Define the objective: maximize the expected return (expected_returns.T @ x).
        M.objective('obj', ObjectiveSense.Maximize, Expr.dot(expected_returns, x))

        # Solve the problem.
        M.solve()

        # Retrieve and print the results if the optimization is successful.
        if M.getProblemStatus() == ProblemStatus.OPTIMAL:
            optimal_weights = x.level()
            expected_return = M.getObjective().level()[0]
            print(f"Optimization successful.")
            print(f"Expected Return: {expected_return:.4f}")
            print(f"Optimal Weights: {np.round(optimal_weights, 4)}")
        else:
            print(f"Optimization failed. Problem status: {M.getProblemStatus()}")


if __name__ == '__main__':
    # N = 3  # Number of assets
    # mu = np.array([0.1, 0.15, 0.08]) # Expected returns
    # Sigma = np.array([ # Covariance matrix
    #     [0.02, 0.01, 0.005],
    #     [0.01, 0.04, 0.01],
    #     [0.005, 0.01, 0.015]
    # ])
    # gamma = 0.15 # Maximum allowed portfolio standard deviation

    # simple_portfolio_optimization(N, mu, Sigma, gamma)

    gurobiModel = GurobiModelData("data/ben_ira_prices.csv")
    input_request = { 
        "apply_black_litterman": False,
        "model_data": gurobiModel.model_data
    }

    mean_variance_optimizer = opt.Optimizer.create("MeanVarianceOptimizer", request = input_request)
    mean_variance_optimizer.run_optimizer()
    # tickers = ["AAPL", "MSFT", "NVDA", "F", "AMZN", "TSLA", "KR", "MKTX", "NFLX", \
    #                         "UAL", "AAL", "AMD", "TMUS", "AZO", "TTWO", "LLY", "AVGO", "DPZ", "ENPH", "WM"]
    # tickers = ["AAPL", "MSFT", "NVDA", "F", "AMZN", "TSLA", "AAL", "UAL", "NFLX"]    
    # request = { "Tickers": tickers, "StartDate": "2010-01-01", "EndDate": "2025-12-18" }
    # marketModel = LiveMarketData(request)
    # data = marketModel.model_data()

    # tracking_error_optimizer = opt.Optimizer.create("TrackingErrorOptimizer", model_data = gurobi_data)
    # tracking_error_optimizer.run_optimizer()

    # trans_cost_opt = opt.Optimizer.create("TransactionCostOptimizer", model_data = gurobi_data)
    # results = trans_cost_opt.run_optimizer()
    # print(results)

# FINAL CONFIGURATIONS FOR PRODUCTION

# STRATEGIES = {
#     'aggressive': {
#         'lambda_diversification': 0.5,
#         'target_ratio': 0.077,
#         'target_eff_n': 20,
#         'description': 'Growth-focused with moderate risk controls',
#         'suitable_for': [
#             'Active equity strategies',
#             'High-conviction portfolios',
#             'Shorter investment horizons'
#         ],
#         'risk_warning': 'High concentration - monitor carefully'
#     },
    
#     'balanced': {
#         'lambda_diversification': 1.0,  
#         'target_ratio': 0.069,
#         'target_eff_n': 40,
#         'description': 'Optimal risk-return balance',
#         'suitable_for': [
#             'Standard institutional mandates',
#             'Balanced equity portfolios',
#             'Default configuration'
#         ],
#         'risk_warning': None
#     },
    
#     'conservative': {
#         'lambda_diversification': 1.5,
#         'target_ratio': 0.063,
#         'target_eff_n': 60,
#         'description': 'Conservative with strong risk controls',
#         'suitable_for': [
#             'Risk-averse mandates',
#             'Regulatory constrained portfolios',
#             'Long-term institutional'
#         ],
#         'risk_warning': None
#     }
# }



# if __name__ == '__main__':
#     # Create Data Environment
#     gurobi_data_model = GurobiModelData()
#     model_data = gurobi_data_model.model_data()
#     rebalance_problem = {
#         "max_iter": 10,
#         "tol": 1e-6,
#         "denominator": 1, # initial guess
#         "min_position": 0.001,
#         "max_position": 0.10,
#         "min_holdings": 20,
#         "lambda_diversification": 0.75
#     }
#     lambda_diversifications_coarse = np.arange(0, 1.1, 0.1)

#     # Create Gurobi environment
#     with gp.Env(empty=True) as env:
#         env.setParam('OutputFlag', 1)
#         env.start()

#         summary_results = []
#         position_results = []
#         for l_div in lambda_diversifications_coarse:
#             rebalance_problem["lambda_diversification"] = l_div

#             with GainLossOptimizer(env, "GainLossOptimizer", rebalance_problem) as optimizer:
#                 optimizer.run(model_data)
#                 result = optimizer.output_results()

#                 # Extract summary statistics
#                 positions = result["positions"]
#                 summary_results.append({
#                     "lambda_diversification": result["lambda_diversification"],
#                     "gain_loss_ratio": result["gain_loss_ratio"],
#                     "portfolio_gain": result["portfolio_gain"],
#                     "portfolio_loss": result["portfolio_loss"],
#                     "num_holdings": (positions["weight"] > 0.001).sum(),
#                     "max_weight": positions["weight"].max(),
#                     "min_weight": positions[positions["weight"] > 0.001]["weight"].min(),
#                     "concentration": (positions["weight"] ** 2).sum(),
#                     "effective_n": 1 / (positions["weight"] ** 2).sum(),
#                     "top5_concentration": positions["weight"].nlargest(5).sum(),
#                     "top10_concentration": positions["weight"].nlargest(10).sum(),
#                 })
                
#                 # Store positions with run identifier
#                 positions_df = positions
#                 positions_df["lambda_diversification"] = l_div
#                 position_results.append(positions_df)
        
#         # Create summary DataFrame
#         summary_df = pd.DataFrame(summary_results)
        
#         # Create positions DataFrame
#         if position_results:
#             positions_df = pd.concat(position_results, ignore_index=True)
#             # Filter to only significant positions for cleaner output
#             positions_df = positions_df[positions_df["weight"] > 0.001]
        
#         # Print detailed analysis
#         print_detailed_analysis(summary_df)
        
#         # Create visualizations
#         create_visualizations(summary_df, output_dir=".")

#         # Save both
#         summary_df.to_csv("parameter_sweep_summary.csv", index=False)
#         positions_df.to_csv("parameter_sweep_positions.csv", index=False)
        
#         # Copy summary to clipboard
#         summary_df.to_clipboard(index=False)
