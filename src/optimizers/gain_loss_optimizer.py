import pandas as pd
import numpy as np
from abc import abstractmethod
import gurobipy as gp
from gurobipy import GRB

class BaseOptimizer:
    def __init__(self, env: gp.Env, model_name: str):
        self.env = env
        self.model_name = model_name
        self.model = None

    def __enter__(self):
        """Enter context manager."""
        self.model = gp.Model(self.model_name, env=self.env)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False
    
    def run(self, model_data: dict):
        self.setup_data(model_data)
        self.build_model()
        self.solve()

    def setup_data(self, model_data: dict):
        self.mu = model_data.get("Mean", None)
        self.price = model_data.get("Prices", None)
        self.returns = model_data.get("Returns", None)
        self.sigma = model_data.get("CovMatrix", None)
        self.num_assets = len(self.mu)
        self.asset_names = self.mu.index.tolist()

    def build_model(self):
        self.setup_decision_variables()
        self.setup_constraints()
        self.setup_objectives()
        self.model.update()

    def solve(self):
        self.model.update()
        self.model.optimize()

    @abstractmethod
    def setup_decision_variables(self): ...

    @abstractmethod
    def setup_constraints(self): ...

    @abstractmethod
    def setup_objectives(self): ...

class GainLossOptimizer(BaseOptimizer):
    def __init__(self, env: gp.Env, model_name: str, rebalance_problem: dict):
        super().__init__(env, model_name)
        self.rebalance_problem = rebalance_problem
        self.max_iter = self.rebalance_problem.get("max_iter", 10)
        self.tol = self.rebalance_problem.get("tol", 1e-6)
        self.denominator = self.rebalance_problem.get("denominator", 1)
        self.min_position = self.rebalance_problem.get("min_position", 0.01)
        self.max_position = self.rebalance_problem.get("max_position", 0.10)
        self.min_holdings = self.rebalance_problem.get("min_holdings", 500)
        self.lambda_diversification = self.rebalance_problem.get("lambda_diversification", 0.75)

    def setup_decision_variables(self):
        self.x = self.model.addMVar(self.num_assets, lb=0, ub=1, name="x")
        self.z = self.model.addMVar(self.num_assets, vtype=GRB.BINARY, name="z")
        self.model.update()

    def setup_constraints(self):
        self._setup_gain_loss_data()
        self._setup_portfolio_constraints()
        self.model.update()

    def _setup_portfolio_constraints(self):
        self.model.addConstr(self.x.sum() == 1, name="portfolio_weight_constr")

        for i in range(self.num_assets):
            self.model.addConstr(self.x[i] <= self.max_position * self.z[i], name=f"max_pos_ub_{i}")
            self.model.addConstr(self.x[i] >= self.min_position * self.z[i], name=f"min_pos_{i}")

    def _setup_gain_loss_data(self):
        positive_returns = self.returns[self.returns > 0] 
        negative_returns = self.returns[self.returns < 0]

        gain = positive_returns.mean().fillna(0)
        loss = abs(negative_returns.mean()).fillna(0)

        self.portfolio_gain = gp.quicksum(
            gain.iloc[i] * self.x[i] for i in range(self.num_assets)
        )
        self.portfolio_loss = gp.quicksum(
            loss.iloc[i] * self.x[i] for i in range(self.num_assets)
        )
        
    def setup_objectives(self):
        pass

    def _compute_pareto_frontier_endpoints(self):
        original_params = self.model.Params.OutputFlag
        self.model.Params.OutputFlag = 0

        gain_loss_obj = (self.portfolio_gain - self.portfolio_loss) / self.denominator
        concentration_obj = gp.quicksum(self.x[i] * self.x[i] for i in range(self.num_assets))           

        self.model.setObjective(gain_loss_obj, GRB.MAXIMIZE)
        self.model.optimize()
        concentration_at_gl_opt = concentration_obj.getValue()

        self.model.setObjective(concentration_obj, GRB.MINIMIZE)
        self.model.optimize()
        concentration_at_concentration_opt = concentration_obj.getValue()

        self.model.Params.OutputFlag = original_params
        self.model.reset()        
        return concentration_at_gl_opt, concentration_at_concentration_opt

    def solve(self):
        """Iteratively optimize gain/loss ratio while adjusting diversification constraint."""   
        concentration_at_gl_opt, concentration_at_concentration_opt = self._compute_pareto_frontier_endpoints()
        self.model.reset()
        self.model.update()
        concentration_threshold = concentration_at_gl_opt + self.lambda_diversification * \
            (concentration_at_concentration_opt - concentration_at_gl_opt)
        concentration_expr = gp.quicksum(self.x[i] * self.x[i] for i in range(self.num_assets))
        self.model.addConstr(concentration_expr <= concentration_threshold, name=f"epsilon_concentration_constr")

        for iteration in range(self.max_iter):
            gain_loss = (self.portfolio_gain - self.portfolio_loss) / self.denominator
            self.model.setObjective(gain_loss, GRB.MAXIMIZE)
            self.model.update()
            self.model.optimize()

            if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                print(f"Failed at iteration {iteration}")
                break                

            g_val = self.portfolio_gain.getValue()
            l_val = self.portfolio_loss.getValue()
            new_denom = g_val + l_val

            if(abs(new_denom - self.denominator) < self.tol):
                break

            self.denominator = new_denom

    def output_results(self) -> pd.DataFrame:
        if self.model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            weights = [self.x[i].X for i in range(self.num_assets)]
            positions = pd.Series(
                data=weights,
                index=self.asset_names,
                name="weight",
                dtype=float
            )
            G_val = self.portfolio_gain.getValue()
            L_val = self.portfolio_loss.getValue()
            ratio = (G_val - L_val) / (G_val + L_val)
            positions = positions.reset_index()
            positions = positions.rename(columns = {"index": "tickers"})

            return {
                "lambda_diversification": self.lambda_diversification,
                "gain_loss_ratio": ratio,
                "portfolio_gain": G_val,
                "portfolio_loss": L_val,
                "positions": positions
            }