import gurobipy as gp
import pandas as pd
import numpy as np
from pathlib import Path
from gurobipy import GRB

from portfolio_optimizers.signals.market_signals import SyntheticSignals, GARCHSignals

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
    def buy_cost(self) -> float:
        return self._config.get("buy_cost", 0.002)

    @property
    def sell_cost(self) -> float:
        return self._config.get("sell_cost", 0.001)

    @property
    def hold_cost(self) -> float:
        return self._config.get("hold_cost", 0.0005)
    
class MultiPeriodOptimizer:
    def __init__(self, env: gp.Env):
        """
        Initialize optimizer with Gurobi environment.
        
        Args:
            env: Gurobi environment (pre-configured)
        """
        self.env = env
        self.model = None

    def __enter__(self):
        """Enter context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False

    def set_data(self, portfolio_configuration: PortfolioConfiguration):
        """Set the market data and configuration for the optimizer."""
        # Configuration
        self.portfolio_configuration = portfolio_configuration

        # Market forecasts
        self.market_data = portfolio_configuration.market_data
        self.mu_levels = portfolio_configuration.mu_levels
        self.sigma_levels = portfolio_configuration.sigma_levels

        # Portfolio state and dimensions
        self.current_weights = portfolio_configuration.current_weights
        self.time_horizon = portfolio_configuration.time_horizon
        self.n_constituents = portfolio_configuration.n_constituents
        self.securities = portfolio_configuration.securities

        # Objective parameters and costs
        self.risk_aversion = portfolio_configuration.risk_aversion
        self.phi_buy_cost = portfolio_configuration.buy_cost
        self.phi_sell_cost = portfolio_configuration.sell_cost
        self.psi_hold_cost = portfolio_configuration.hold_cost

    def build_model(self):
        """Build the optimization model"""
        self.model = gp.Model("MultiPeriodOptimizer", env=self.env)
        self.model.Params.NonConvex = 2
        self._create_decision_variables()
        self._setup_constraints()
        self._set_objective()
        self.model.update()

    def solve(self) -> str:
        """Solve the optimization model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            self._extract_solution()
            return "OPTIMAL"
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            self.model.computeIIS()
            self.model.write("long_short_portfolio.ilp")
            return "INFEASIBLE"
        else:
            return f"STATUS_{self.model.status}"
        
    def _create_decision_variables(self): 
        """Create the decision variables"""
        self.trades = self.model.addVars(range(1, self.time_horizon+1), range(self.n_constituents), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="trades")
        self.buy_trades = self.model.addVars(range(1, self.time_horizon+1), range(self.n_constituents), vtype=GRB.CONTINUOUS, lb=0.0, name="buy_trades")
        self.sell_trades = self.model.addVars(range(1, self.time_horizon+1), range(self.n_constituents), vtype=GRB.CONTINUOUS, lb=0.0, name="sell_trades")
        self.is_buying = self.model.addVars(range(1, self.time_horizon+1), range(self.n_constituents), vtype=gp.GRB.BINARY, name="is_buying")
        self.is_selling = self.model.addVars(range(1, self.time_horizon+1), range(self.n_constituents), vtype=gp.GRB.BINARY, name="is_selling")        
        self.optimal_weights = self.model.addMVar((self.time_horizon+1, self.n_constituents), vtype=GRB.CONTINUOUS, lb=0.0, name="optimal_weights")

    def _setup_constraints(self):
        """Set up the constraints for the optimization model"""
        for n in range(self.n_constituents):
            self.model.addConstr(
                self.optimal_weights[0, n].item() == self.current_weights[n], 
                name=f"initial_state{n}"
            )

        for t in range(self.time_horizon + 1):
            self.model.addConstr(
                gp.quicksum(self.optimal_weights[t, n].item() for n in range(self.n_constituents)) == 1.0,
                name=f"net_exposure"
            )
        
        for t in range(self.time_horizon):
            t_next = t + 1

            for n in range(self.n_constituents):
                self.model.addConstr(
                    self.is_buying[t_next, n] + self.is_selling[t_next, n] <= 1.0,
                    name=f"mutually_exlcusive_guard_t_{t}_asset_{n}"
                )

                self.model.addGenConstrIndicator(
                    self.is_buying[t_next, n], 0, self.buy_trades[t_next, n], GRB.EQUAL, 0.0,
                    name=f"indicator_buy_t_{t}_asset{n}"
                )

                self.model.addGenConstrIndicator(
                    self.is_selling[t_next, n], 0, self.sell_trades[t_next, n], GRB.EQUAL, 0.0,
                    name=f"indicator_sell_t_{t}_asset{n}"
                )
                
                self.model.addConstr(
                    self.trades[t_next, n] == self.buy_trades[t_next, n] - self.sell_trades[t_next, n],
                    name=f"total_trade_t_{t}_asset{n}"
                )

                self.model.addConstr(
                    self.optimal_weights[t_next, n].item() == self.optimal_weights[t, n].item() + self.trades[t_next, n],
                    name=f"trade_def_t_{t}_asset{n}"
                )

        # terminal_weights = [1.0 / self.n_constituents] * self.n_constituents # x_H target allocation
        # for n in range(self.n_constituents):
        #     self.model.addConstr(self.optimal_weights[self.time_horizon, n].item() == terminal_weights[n], name=f"terminal_state_{n}")

    def _set_objective(self) -> pd.DataFrame:
        """Set the optimizer objective"""
        objective = gp.QuadExpr()

        for t in range(self.time_horizon):
            # get the next period index
            t_next = t + 1

            # Get the optimal weights for the next period
            w_t = self.optimal_weights[t_next, :]

            # expected return
            return_term = w_t @ np.array(self.mu_levels[t])

            # quadratic risk term
            risk_term = w_t @ np.array(self.sigma_levels[t]) @ w_t

            # transaction cost
            tc_term = gp.quicksum((self.phi_buy_cost * self.buy_trades[t_next, n]) + \
                                  (self.phi_sell_cost * self.sell_trades[t_next, n]) for n in range(self.n_constituents))

            # holding cost
            holding_term = gp.quicksum(self.psi_hold_cost * self.optimal_weights[t_next, n] for n in range(self.n_constituents))

            # Combine terms into the objective function
            objective += return_term - (self.risk_aversion * risk_term) - tc_term - holding_term

        self.model.setObjective(objective, GRB.MAXIMIZE)

    @staticmethod
    def _solution_value(value) -> float:
        return float(np.asarray(value).item())

    def _extract_solution(self): 
        """Extract multi-period solutions over the time timeline."""

        records = []
        for t in range(1, self.time_horizon + 1):
            t_forecast = t - 1

            period_weights = np.array([
                self._solution_value(self.optimal_weights[t, n].X)
                for n in range(self.n_constituents)
            ])
            period_mu = np.array(self.mu_levels[t_forecast])
            period_return_t = self._solution_value(period_weights @ period_mu)

            n_positions = len([w for w in period_weights if w > 1e-5])
            for n in range(self.n_constituents):
                records.append({
                    "Period": t,
                    "Security": self.securities[n],
                    "Weight": period_weights[n],
                    "Buy_Trade": self._solution_value(self.buy_trades[t, n].X),
                    "Sell_Trade": self._solution_value(self.sell_trades[t, n].X),
                    "Net_Trade": self._solution_value(self.trades[t, n].X),
                    "Risk_Aversion": self.risk_aversion,
                    "Period_Return": period_return_t,
                    "Active_Positions": n_positions
                })

        self.solution_df = pd.DataFrame(records)

    def get_solution(self) -> dict:
        if self.optimal_weights is None:
            raise ValueError("No solution available. Call solve() first.")

        return {
            'solution': self.solution_df,
            'objective_value': self.model.ObjVal
        }