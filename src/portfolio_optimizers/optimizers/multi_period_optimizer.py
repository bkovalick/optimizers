import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from portfolio_optimizers.optimizers.configuration import PortfolioConfiguration

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
        self.terminal_weights = portfolio_configuration.terminal_weights
        self.current_weights = portfolio_configuration.current_weights
        self.time_horizon = portfolio_configuration.time_horizon
        self.n_constituents = portfolio_configuration.n_constituents
        self.securities = portfolio_configuration.securities

        # Objective parameters and costs
        self.risk_aversion = portfolio_configuration.risk_aversion
        self.phi_buy_costs = portfolio_configuration.buy_costs
        self.phi_sell_costs = portfolio_configuration.sell_costs
        self.psi_hold_cost = portfolio_configuration.hold_cost
        self.period_turnover_limit = portfolio_configuration.period_turnover_limit
        self.global_horizon_turnover_limit = portfolio_configuration.global_horizon_turnover_limit
        self.net_target = portfolio_configuration.net_target
        self.gross_exposure = portfolio_configuration.gross_exposure
        self.max_long = portfolio_configuration.max_long
        self.max_short = portfolio_configuration.max_short
        self.max_position = portfolio_configuration.max_position

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
        """Create decision variables for the optimization model."""
        self.trades = self.model.addVars(
            range(1, self.time_horizon + 1), 
            range(self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=-GRB.INFINITY, 
            name="trades"
        )

        self.buy_trades = self.model.addVars(
            range(1, self.time_horizon + 1), 
            range(self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name="buy_trades"
        )

        self.sell_trades = self.model.addVars(
            range(1, self.time_horizon + 1), 
            range(self.n_constituents), 
            vtype=GRB.CONTINUOUS, lb=0.0, 
            name="sell_trades"
        )

        self.is_buying = self.model.addVars(
            range(1, self.time_horizon + 1), 
            range(self.n_constituents), 
            vtype=gp.GRB.BINARY, 
            name="is_buying")
        
        self.is_selling = self.model.addVars(
            range(1, self.time_horizon + 1), 
            range(self.n_constituents), 
            vtype=gp.GRB.BINARY, 
            name="is_selling")
        
        self.optimal_weights = self.model.addMVar(
            (self.time_horizon + 1, self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=-self.max_position, 
            ub=self.max_position,
            name="optimal_weights"
        )

        self.abs_optimal_weights = self.model.addMVar(
            (self.time_horizon + 1, self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name="abs_optimal_weights"
        )

        self.optimal_long_weights = self.model.addMVar(
            (self.time_horizon + 1, self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name="optimal_long_weights"
        )  

        self.optimal_short_weights = self.model.addMVar(
            (self.time_horizon + 1, self.n_constituents), 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name="optimal_short_weights"
        )

    def _setup_constraints(self):
        """Set up the constraints for the optimization model"""
        self._setup_portfolio_constraints()
        self._setup_gross_exposure_constraints()
        self._setup_long_short_decomposition_constraints()
        self._setup_turnover_constraints()
        self._setup_trade_indicator_constraints()
                
    def _setup_portfolio_constraints(self):
        """Set up portfolio constraints for the optimization model"""
        for n in range(self.n_constituents):
            self.model.addConstr(
                self.optimal_weights[0, n].item() == self.current_weights[n], 
                name=f"initial_state{n}"
            )

        for t in range(self.time_horizon + 1):
            self.model.addConstr(
                gp.quicksum(self.optimal_weights[t, n].item() for n in range(self.n_constituents)) == self.net_target,
                name=f"net_exposure_t_{t}"
            )      
            
        for t in range(1, self.time_horizon + 1):
            for n in range(self.n_constituents):
                self.model.addConstr(
                    self.trades[t, n] == self.buy_trades[t, n] - self.sell_trades[t, n],
                    name=f"total_trade_t_{t}_asset{n}"
                )

                self.model.addConstr(
                    self.optimal_weights[t, n].item() == self.optimal_weights[t - 1, n].item() + self.trades[t, n],
                    name=f"trade_def_t_{t}_asset{n}"
                )

        # for n in range(self.n_constituents):
        #     self.model.addConstr(
        #         self.optimal_weights[self.time_horizon, n].item() == self.terminal_weights[n], 
        #         name=f"terminal_state_{n}"
        #     )

    def _setup_gross_exposure_constraints(self):
        """Set up gross exposure constraints for the optimization model"""
        for t in range(1, self.time_horizon + 1):
            self.model.addConstr(
                gp.quicksum(
                    self.optimal_long_weights[t, n].item() + self.optimal_short_weights[t, n].item()
                    for n in range(self.n_constituents)
                ) <= self.gross_exposure,
                name=f"gross_exposure_t_{t}"
            )

    def _setup_long_short_decomposition_constraints(self):
        """Set up long-short decomposition constraints for the optimization model"""
        for t in range(1, self.time_horizon + 1):
            for n in range(self.n_constituents):
                self.model.addConstr(
                    self.optimal_long_weights[t, n].item() ==
                    0.5 * (self.abs_optimal_weights[t, n].item() + self.optimal_weights[t, n].item()),
                    name=f"long_def_t_{t}_asset{n}"
                )

                self.model.addConstr(
                    self.optimal_short_weights[t, n].item() ==
                    0.5 * (self.abs_optimal_weights[t, n].item() - self.optimal_weights[t, n].item()),
                    name=f"short_def_t_{t}_asset{n}"
                )

                self.model.addConstr(
                    self.optimal_long_weights[t, n].item() <= self.max_long,
                    name=f"max_long_t_{t}_asset{n}"
                )
                
                self.model.addConstr(
                    self.optimal_short_weights[t, n].item() <= self.max_short,
                    name=f"max_short_t_{t}_asset{n}"
                )

    def _setup_turnover_constraints(self):
        """Set up turnover constraints for the optimization model"""
        for t in range(1, self.time_horizon + 1):
            self.model.addConstr(
                0.5 * gp.quicksum(self.buy_trades[t, n] + self.sell_trades[t, n] 
                for n in range(self.n_constituents)) <= self.period_turnover_limit,
                name=f"period_turnover_cap_t_{t}"
            )

        self.model.addConstr(
            0.5 * gp.quicksum(
                self.buy_trades[t, n] + self.sell_trades[t, n]
                for t in range(1, self.time_horizon + 1)
                for n in range(self.n_constituents)
            ) <= self.global_horizon_turnover_limit,
            name=f"global_horizon_turnover_cap_t_{self.time_horizon}"
        )

    def _setup_trade_indicator_constraints(self):
        """Set up trade indicator constraints for the optimization model"""
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
            tc_term = gp.quicksum((self.phi_buy_costs[t] * self.buy_trades[t_next, n]) + \
                                  (self.phi_sell_costs[t] * self.sell_trades[t_next, n]) for n in range(self.n_constituents))

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
            period_long_weights = np.array([
                self._solution_value(self.optimal_long_weights[t, n].X)
                for n in range(self.n_constituents)
            ])
            period_short_weights = np.array([
                self._solution_value(self.optimal_short_weights[t, n].X)
                for n in range(self.n_constituents)
            ])
            period_net_trades = np.array([
                self._solution_value(self.trades[t, n].X)
                for n in range(self.n_constituents)
            ])
            period_turnover_pct = 0.5 * np.abs(period_net_trades).sum() * 100
            period_mu = np.array(self.mu_levels[t_forecast])
            period_return_t = self._solution_value(period_weights @ period_mu)

            n_positions = np.sum(np.abs(period_weights) > 1e-5)
            for n in range(self.n_constituents):
                records.append({
                    "Period": t,
                    "Security": self.securities[n],
                    "Weight": period_weights[n],
                    "Long_Weight": period_long_weights[n],
                    "Short_Weight": period_short_weights[n],
                    "Buy_Trade": self._solution_value(self.buy_trades[t, n].X),
                    "Sell_Trade": self._solution_value(self.sell_trades[t, n].X),
                    "Net_Trade": period_net_trades[n],
                    "Risk_Aversion": self.risk_aversion,
                    "Period_Return": period_return_t,
                    "Active_Positions": n_positions,
                    "Turnover_Pct": period_turnover_pct
                })

        self.solution_df = pd.DataFrame(records)
        self.solution_df["Max_Time_Horizon"] = self.time_horizon
        self.solution_df["Objective_Value"] = self.model.ObjVal

    def get_solution(self) -> dict:
        if self.optimal_weights is None:
            raise ValueError("No solution available. Call solve() first.")

        return {
            'solution': self.solution_df,
            'objective_value': self.model.ObjVal
        }