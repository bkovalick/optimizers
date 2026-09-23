import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime

class RebalanceProblem:
    def __init__(self, config: dict):
        self.config = config

    @property
    def tickers(self) -> list:
        return self.config.get("tickers", [])

    @property
    def native_tickers(self) -> list:
        return self.config.get("native_tickers", [])

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def target_weights(self) -> np.ndarray:
        return np.array([
            1 / len(self.native_tickers) if t in self.tickers else 0 
                for t in self.tickers
        ])
    
    @property
    def tax_preference(self) -> float:
        return self.config.get("tax_preference", 0.0)
    
    @tax_preference.setter
    def tax_preference(self, value: float):
        self.config["tax_preference"] = value

    @property
    def total_gains_budget(self) -> dict:
        return self.config.get("total_gains_budget", {})
    
    @property
    def short_term_gains_budget(self) -> dict:
        return self.config.get("short_term_gains_budget", {})    

    @ property
    def rebalance_date(self) -> str:
        return self.config.get("rebalance_date", "")

    @property
    def portfolio_parameters(self) -> dict:
        return self.config.get("portfolio_parameters", None)

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        return self.config.get("covariance_matrix", None)

    @property
    def tax_lots(self) -> pd.DataFrame:
        return self.config.get("tax_lots", None)
    
    @property
    def n_lots(self) -> int:
        return len(self.tax_lots) if self.tax_lots is not None else 0

    @property
    def total_portfolio_value(self) -> float:
        return self.config.get("total_portfolio_value", 0.0)
    
    @property
    def initial_amounts(self) -> np.ndarray:
        return self.config.get("initial_amounts", np.zeros(len(self.tickers)))

    @property
    def initial_weights(self) -> np.ndarray:
        return self.config.get("initial_weights", np.zeros(len(self.tickers)))
    
class RebalanceProblemBuilder:

    def __init__(self, config: dict):
        self.config = config
        self.daily_price_file = self.config.get("daily_price_file", "")
        if self.daily_price_file != "":
            self.price_data = pd.read_csv(self.daily_price_file, parse_dates=["Price_Date"], index_col="Price_Date")
        else:
            raise ValueError("Please provide a valid input file.")
        self.tickers = self.config.get("tickers", [])
        self.native_tickers = self.config.get("native_tickers", [])
        self.portfolio_parameters = self.config.get("portfolio_parameters", {})
        self.rebalance_date = self.config.get("rebalance_date", "")
        self.tax_lots = self._build_tax_lots()
        self.covariance_matrix = self._build_covariance_matrix()

    def build(self) -> RebalanceProblem:
        initial_amounts = self.tax_lots.groupby("Ticker")["CurrentValue"].sum()
        initial_amounts = initial_amounts.reindex(self.tickers).fillna(0)
        total_portfolio_value = initial_amounts.sum()
        initial_weights = initial_amounts / total_portfolio_value if total_portfolio_value > 0 else np.zeros(len(self.tickers))
        initial_weights = initial_weights.values
        rebalance_problem = {
            "tickers": self.tickers,
            "native_tickers": self.native_tickers,
            "portfolio_parameters": self.portfolio_parameters,
            "covariance_matrix": self.covariance_matrix,
            "tax_lots": self.tax_lots,
            "rebalance_date": self.rebalance_date,
            "initial_amounts": initial_amounts,
            "initial_weights": initial_weights,
            "total_portfolio_value": total_portfolio_value,
            "total_gains_budget": self.config.get("total_gains_budget", {}),
            "short_term_gains_budget": self.config.get("short_term_gains_budget", {})
        }
        return RebalanceProblem(rebalance_problem)
    
    @property
    def prices(self) -> pd.DataFrame:
        return self.price_data[self.tickers]

    def _build_covariance_matrix(self) -> pd.DataFrame:
        returns = self.prices.pct_change().replace([np.inf, -np.inf], np.nan)
        returns = returns[(returns > -1) & (returns < 1e6)]  # filter out -100% and absurd spikes
        returns = returns.dropna(how='any')  # drop rows with any NaN
        cov = returns.cov() * 252
        return cov

    def _build_tax_lots(self) -> pd.DataFrame:
        tax_lots = {}
        for deposit_date, deposit_info in self.portfolio_parameters.items():
            deposit_amount = deposit_info["amount"]
            allocations = deposit_info["weights"]
            for ticker in self.tickers:
                alloc_by_ticker = allocations.get(ticker, None)
                if alloc_by_ticker is None:
                    continue
                
                deposit_date_dt = datetime.strptime(deposit_date, "%Y-%m-%d")
                rebalance_date_dt = datetime.strptime(self.rebalance_date, "%Y-%m-%d")
                deposit_by_ticker = alloc_by_ticker * deposit_amount
                current_price = self.prices.loc[rebalance_date_dt, ticker]
                acq_price = self.prices.loc[deposit_date_dt, ticker]
                term = "ST" if (rebalance_date_dt - deposit_date_dt).days <= 365 else "LT"
                tax_rate = 0.37 if term == "ST" else 0.20 
                shares = deposit_by_ticker / acq_price
                current_value = shares * current_price
                gain_per_share = current_price - acq_price
                total_gain = gain_per_share * shares
                max_tax_cost = total_gain * tax_rate
                tax_lots[(deposit_date, ticker)] = {
                    "CurrentPrice": current_price,
                    "AcqPrice": acq_price,
                    "Term": term,
                    "TaxRate": tax_rate,
                    "CurrentValue": current_value,
                    "Shares": shares,
                    "GainPerShare": gain_per_share,
                    "TotalGain": total_gain,
                    "MaxTaxCost": max_tax_cost
                }

        tax_lots_df = pd.DataFrame.from_dict(tax_lots, orient="index")
        tax_lots_df.index.names = ["DepositDate", "Ticker"]
        tax_lots_df = tax_lots_df.reset_index()
        return tax_lots_df

class TaxLotOptimizer:

    def optimize(self, 
                 rebalance_problem: RebalanceProblem):
        decision_variables = self._setup_decision_variables(rebalance_problem)
        constraints = self._setup_constraints(rebalance_problem, decision_variables)
        tax_preference = rebalance_problem.tax_preference
        if tax_preference != 0 and tax_preference != 1:
            constraints.extend(
                self._setup_epsilon_constraint(rebalance_problem, decision_variables)
            )
        objective = self._setup_epsilon_objective(rebalance_problem, decision_variables)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(cp.CLARABEL, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError("Optimizer failed")
        
        return self._process_solution(rebalance_problem, decision_variables)

    def _process_solution(self,
                          rebalance_problem: RebalanceProblem,
                          decision_variables: dict) -> pd.DataFrame:
        total_portfolio_value = rebalance_problem.total_portfolio_value
        tax_lots = rebalance_problem.tax_lots
        rebalanced_weights = decision_variables["rebalanced_weights"].value
        sell_fractions = decision_variables["sell_fractions"].value
        final_amounts = rebalanced_weights * total_portfolio_value

        lot_to_ticker = self._build_lot_matrix(rebalance_problem)
        max_tax_per_ticker = tax_lots.groupby("Ticker")["MaxTaxCost"].sum().reindex(
            rebalance_problem.tickers, fill_value=0
        )
        realized_tax_per_lot = tax_lots["MaxTaxCost"].values * sell_fractions
        tax_cost_per_ticker = lot_to_ticker.T @ realized_tax_per_lot 

        total_gains_per_lot = tax_lots["TotalGain"].values * sell_fractions
        total_gains_realized = lot_to_ticker.T @ total_gains_per_lot

        st_mask = (tax_lots["Term"] == "ST").values
        short_term_gains_per_lot = np.where(st_mask ,tax_lots["TotalGain"].values, 0) * sell_fractions
        short_term_gains_realized = lot_to_ticker.T @ short_term_gains_per_lot

        return pd.DataFrame({
            "Ticker": rebalance_problem.tickers,
            "InitialAmount": rebalance_problem.initial_amounts,
            "TradeAmount": final_amounts - rebalance_problem.initial_amounts,
            "FinalAmount": final_amounts,            
            "InitialWeight": rebalance_problem.initial_weights,
            "RebalancedWeight": rebalanced_weights,
            "WeightChange": rebalanced_weights - rebalance_problem.initial_weights,
            "MaxTaxCost": max_tax_per_ticker.values,
            "RealizedTax": tax_cost_per_ticker,
            "TotalGainsRealized": total_gains_realized,
            "ShortTermGainsRealized": short_term_gains_realized
        })

    def _setup_decision_variables(self,
                                  rebalance_problem: RebalanceProblem) -> dict:
        n_assets = rebalance_problem.n_assets
        n_lots = rebalance_problem.n_lots
        return {
            "rebalanced_weights": cp.Variable(n_assets, nonneg=True),
            "buy_trades": cp.Variable(n_assets, nonneg=True),
            "sell_trades": cp.Variable(n_assets, nonneg=True),
            "sell_fractions": cp.Variable(n_lots, nonneg=True)
        }

    def _setup_constraints(self, 
                           rebalance_problem: RebalanceProblem,
                           decision_variables: dict) -> list:
        constraints = []
        constraints.extend(
            self._setup_portfolio_constraints(rebalance_problem, decision_variables)
        )
        constraints.extend(
            self._setup_tax_constraints(rebalance_problem, decision_variables)
        )
        constraints.extend(
            self._setup_net_buy_sell_restrictions(rebalance_problem, decision_variables)
        )
        constraints.extend(
            self._setup_gains_budgets(rebalance_problem, decision_variables)
        )
        return constraints
    
    def _setup_portfolio_constraints(self, 
                                     rebalance_problem: RebalanceProblem,
                                     decision_variables: dict):
        current_weights = rebalance_problem.initial_weights
        rebalanced_weights = decision_variables["rebalanced_weights"]
        buy_trades = decision_variables["buy_trades"]
        sell_trades = decision_variables["sell_trades"]    
        return [
                cp.sum(rebalanced_weights) == 1,
                rebalanced_weights == current_weights + buy_trades - sell_trades,
                cp.sum(sell_trades) == cp.sum(buy_trades)
            ]
        
    def _setup_tax_constraints(self, 
                               rebalance_problem: RebalanceProblem,
                               decision_variables: dict):
        tax_lots = rebalance_problem.tax_lots
        total_portfolio_value = rebalance_problem.total_portfolio_value
        sell_trades = decision_variables["sell_trades"]
        sell_fractions = decision_variables["sell_fractions"]

        lot_to_ticker = self._build_lot_matrix(rebalance_problem)
        dollar_sold_per_lot = cp.multiply(tax_lots["CurrentValue"].values, sell_fractions)
        weight_reduction = lot_to_ticker.T @ (dollar_sold_per_lot / total_portfolio_value)
        return [
                sell_fractions <= 1,
                sell_trades == weight_reduction
            ]       

    def _setup_epsilon_constraint(self, 
                                  rebalance_problem: RebalanceProblem,
                                  decision_variables: dict): 
        total_portfolio_value = rebalance_problem.total_portfolio_value
        tax_preference = rebalance_problem.tax_preference

        # 1. Portfolio-optimal
        port_vars = self._setup_decision_variables(rebalance_problem)
        port_constraints = self._setup_constraints(rebalance_problem, port_vars)
        port_te_obj = self._setup_portfolio_objective(rebalance_problem, port_vars)
        port_tax_obj = self._setup_tax_objective(rebalance_problem, port_vars)
        port_problem = cp.Problem(cp.Minimize(port_te_obj), port_constraints)
        port_problem.solve(solver = cp.CLARABEL, verbose=False)
        tax_at_port_opt = port_tax_obj.value / total_portfolio_value
        
        # 2. Tax-optimal
        tax_vars = self._setup_decision_variables(rebalance_problem)
        tax_constraints = self._setup_constraints(rebalance_problem, tax_vars)  
        tax_obj = self._setup_tax_objective(rebalance_problem, tax_vars)      
        tax_problem = cp.Problem(cp.Minimize(tax_obj), tax_constraints)
        tax_problem.solve(solver = cp.CLARABEL, verbose=False)
        tax_at_tax_opt = tax_obj.value / total_portfolio_value   

        # 3. Interpolate Threshold
        tax_threshold = tax_at_port_opt + tax_preference * (tax_at_tax_opt - tax_at_port_opt)

        # 4. Add epsilon constraint
        reset_tax_obj = self._setup_tax_objective(rebalance_problem, decision_variables)
        epsilon_constraint = [reset_tax_obj / total_portfolio_value <= tax_threshold]
        return epsilon_constraint
    
    def _setup_gains_budgets(self, 
                             rebalance_problem: RebalanceProblem,
                             decision_variables: dict):
        tax_lots = rebalance_problem.tax_lots
        total_gains_budget = rebalance_problem.total_gains_budget
        short_term_gains_budget = rebalance_problem.short_term_gains_budget
        sell_fractions = decision_variables["sell_fractions"]
        st_gains_lots = np.where(tax_lots["Term"] == "ST", tax_lots["TotalGain"].values, 0)
        total_gains_lots = tax_lots["TotalGain"].values

        constraints = []
        if total_gains_budget["apply_budget"]:
            tg_amount = total_gains_budget["amount"]
            constraints.append(cp.sum(cp.multiply(total_gains_lots, sell_fractions)) <= tg_amount)

        if short_term_gains_budget["apply_budget"]:
            st_amount = short_term_gains_budget["amount"]
            constraints.append(cp.sum(cp.multiply(st_gains_lots, sell_fractions)) <= st_amount)

        return constraints
    
    def _setup_net_buy_sell_restrictions(self,
                                         rebalance_problem: RebalanceProblem,
                                         decision_variables: dict):
        initial_weights = rebalance_problem.initial_weights
        target_weights = rebalance_problem.target_weights
        sell_fractions = decision_variables["sell_fractions"]

        constraints = []
        lot_to_ticker = self._build_lot_matrix(rebalance_problem)
        underweight_mask = initial_weights < target_weights
        constraints.append(cp.multiply(sell_fractions, lot_to_ticker.T @ underweight_mask) == 0)
        return constraints
    
    def _setup_epsilon_objective(self,
                         rebalance_problem: RebalanceProblem,
                         decision_variables: dict):
        tax_preference = rebalance_problem.tax_preference
        objective = None
        if tax_preference == 1:
            objective = self._setup_tax_objective(rebalance_problem, decision_variables)
        else:
            objective = self._setup_portfolio_objective(rebalance_problem, decision_variables)
        return objective
    
    def _setup_weighted_sum_objective(self,
                                      rebalance_problem: RebalanceProblem,
                                      decision_variables: dict):
        tax_preference = rebalance_problem.tax_preference
        te_obj = self._setup_portfolio_objective(rebalance_problem, decision_variables)
        tax_obj = self._setup_tax_objective(rebalance_problem, decision_variables)
        objective = tax_preference * te_obj + tax_obj
        return objective
    
    def _setup_portfolio_objective(self, 
                                   rebalance_problem: RebalanceProblem,
                                   decision_variables: dict): 
        target_weights = rebalance_problem.target_weights
        covariance_matrix = rebalance_problem.covariance_matrix.values
        rebalanced_weights = decision_variables["rebalanced_weights"]
        target_deviation = rebalanced_weights - target_weights
        return cp.quad_form(target_deviation, covariance_matrix)

    def _setup_tax_objective(self, 
                             rebalance_problem: RebalanceProblem,
                             decision_variables: dict): 
        tax_lots = rebalance_problem.tax_lots
        sell_fractions = decision_variables["sell_fractions"]
        return cp.sum(cp.multiply(tax_lots["MaxTaxCost"].values, sell_fractions))
    
    def _build_lot_matrix(self, 
                          rebalance_problem: RebalanceProblem):
        tickers = rebalance_problem.tickers
        tax_lots = rebalance_problem.tax_lots

        n_rows = len(tax_lots)
        n_cols = len(tickers)
        lot_matrix = np.zeros((n_rows, n_cols))
        for i, ticker in enumerate(tax_lots["Ticker"].values):
            j = tickers.index(ticker)
            lot_matrix[i][j] = 1
        return lot_matrix

class OptimizationOrchestrator:
    
    def __init__(self, rebalance_problem: RebalanceProblem):
        self.rebalance_problem = rebalance_problem
        self.optimizer = TaxLotOptimizer()

    def run_optimization(self):
        results = self.optimizer.optimize(self.rebalance_problem)
        return results

if __name__ == '__main__':
    rebal_config = {
        "daily_price_file": "data/daily_prices.csv",
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