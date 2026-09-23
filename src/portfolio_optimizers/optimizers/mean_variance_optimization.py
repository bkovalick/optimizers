import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import inspect
import cvxpy as cp

from portfolio_optimizers.signals.black_litterman_signal import BlackLittermanSignal

class DecisionVariables:
    def __init__(self, model: gp.Model, mu: pd.DataFrame):
        # Add variables: x[i] denotes the proportion invested in stock i
        self.x = model.addMVar(len(mu), lb=0, ub=1, name="x")
        # Add variables: x_plus[i] denotes the proportion of stock i bought
        self.x_plus = model.addMVar(len(mu), lb=0, ub=1, name="x_plus")
        # Add variables: x_minus[i] denotes the proportion of stock i sold
        self.x_minus = model.addMVar(len(mu), lb=0, ub=1, name="x_minus")
        # Add variables: b_plus[i]=1 if stock i is bought, and b_plus[i]=0 otherwise
        self.b_plus = model.addMVar(len(mu), vtype=gp.GRB.BINARY, name="b_plus")
        # Add variables: b_minus[i]=1 if stock i is sold, and b_minus[i]=0 otherwise
        self.b_minus = model.addMVar(len(mu), vtype=gp.GRB.BINARY, name="b_minus")
        model.update()

class Optimizer(ABC):
    """Abstract base class that automatically registers its subclasses."""
    _registry = {}

    def __init__(self):
        self.model = None
        self.env = None

    def __enter__(self):
        return self        

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            self.model.dispose()
        return False
    
    def __init_subclass__(cls, opt_type, **kwargs):
        """
        This class method is called every time a class inherits from Optimizer.
        It registers the subclass in the _registry dictionary using the 
        'opt_type' keyword argument passed during class definition.
        """
        super().__init_subclass__(**kwargs)
        if opt_type:
            cls._registry[opt_type] = cls

    @classmethod
    def create(cls, optimizer_type, **kwargs):
        """Factory method to create a subclass instance."""
        subclass = cls._registry.get(optimizer_type)
        if not subclass:
            raise ValueError(f"Unknown vehicle type: {optimizer_type}")

        sig = inspect.signature(subclass.__init__)
        required_args = set(sig.parameters.keys()) - {'self'}
        if not required_args.issubset(kwargs.keys()):
            raise TypeError(f"{subclass.__name__}.__init__ requires arguments: {required_args}")

        return subclass(**kwargs)

    @abstractmethod
    def run_optimizer(self):
        raise NotImplementedError

class MixedIntegerOptimizer(Optimizer, opt_type = "MixedIntegerOptimizer"):
    def __init__(self):
        super().__init__()

    def run_optimizer(self):
        # Create an empty optimization model
        with gp.Env() as env, gp.Model("Mixed Integer Optimizer", env = env) as m:
            # x is a binary variable (0 or 1)
            x = m.addVar(vtype=GRB.BINARY, name="x")
            # y is a continuous variable
            y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
            # z is an integer variable
            z = m.addVar(vtype=GRB.INTEGER, name="z")
            
            # Add constraints
            # Constraint 1: x + 2y + z <= 4
            m.addConstr(x + 2 * y + z <= 4, "c0")
            # Constraint 2: 2z + y <= 5
            m.addConstr(2 * z + y <= 5, "c1")
            # Constraint 3: x + y >= 1
            m.addConstr(x + y >= 1, "c2")

            # Set objective function: Maximize 2x + y + 3z
            m.setObjective(2 * x + y + 3 * z, GRB.MAXIMIZE)

            # Optimize the model
            m.optimize()

            # Print results if an optimal solution is found
            if m.status == GRB.OPTIMAL:
                print("\nOptimal solution found:")
                print(f"x = {x.X}")
                print(f"y = {y.X}")
                print(f"z = {z.X}")
                print(f"Optimal objective value = {m.ObjVal}")
            elif m.status == GRB.INFEASIBLE:
                print("Model is infeasible.")
            elif m.status == GRB.UNBOUNDED:
                print("Model is unbounded.")        

class MeanVarianceOptimizer(Optimizer, opt_type = "MeanVarianceOptimizer"):
    def __init__(self,
                 request: dict):
        super().__init__()

        self.request = request
        self.model_data = self.request.get("model_data", None)
        if self.model_data is None:
            raise ValueError("Must provide model data to initialize MeanVarianceOptimizer")
        
        self.num_assets = len(self.model_data.get("Mean", []))
        self.returns = self.model_data.get("Mean", None)
        self.prices = self.model_data.get("Prices", None)
        self.apply_black_litterman = self.request.get("apply_black_litterman", False)
        self.market_caps = self.request.get("market_caps", None)

    def run_optimizer(self):
        # Create an empty optimization model
        with gp.Env() as env, gp.Model("Mixed Integer Optimizer", env = env) as model:
            model.setParam("MIPGap", 0.01)

            # Data Parameters
            sigma = self.model_data.get("CovMatrix", None)
            if sigma is None:
                raise ValueError("Must provide covariance matrix in order to utilize Black-Litterman Signal")

            mu = self.model_data.get("Mean", None)
            # mu = self.model_data.get("Mean", None) \
            #     if not self.apply_black_litterman else self._apply_black_litterman(sigma)

            # Add variables: x[i] denotes thez proportion invested in stock i, 0 <= x[i] <= 1
            x = model.addMVar(self.num_assets, lb=0, ub=1, name="x")

            # Add variables: b[i]=1 if stock i is held, and b[i]=0 otherwise
            b = model.addMVar(self.num_assets, vtype=gp.GRB.BINARY, name="b")

            risk_constr = self._setup_constraints(model, x, b, sigma)

            # Define objective function: Maximize expected return
            model.setObjective(mu.to_numpy() @ x, gp.GRB.MAXIMIZE)

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                print(f"Expected return:  {model.ObjVal:.6f}")
                print(f"Variance:         {x.X @ sigma @ x.X:.6f}")
                print(f"Solution time:    {model.Runtime:.2f} seconds\n")

                # Print investments (with non-negligible values, i.e., > 1e-5)
                trade_flags = pd.Series(name="TradeFlags", data=b.X, index=mu.index)
                positions = pd.Series(name="Position", data=x.X, index=mu.index)
                print(f"Number of trades: {positions[positions > 1e-5].count()}\n")
                print(positions[positions > 1e-5])
                print(trade_flags)

                risks = np.linspace(1.37, 4, 20)
                returns = np.zeros(risks.shape)
                npos = np.zeros(risks.shape)

                # hide Gurobi log output
                model.params.OutputFlag = 0

                # solve the model for each risk level
                for i, risk_level in enumerate(risks):
                    # set risk level: RHS of risk constraint
                    risk_constr.QCRHS = risk_level**2
                    model.optimize()
                    # store data
                    returns[i] = mu @ x.X
                    npos[i] = len(x.X[x.X > 1e-5])

                fig, axs = plt.subplots(1, 2, figsize=(10, 3))

                # Axis 0: The efficient frontier
                axs[0].scatter(x=risks, y=returns, marker="o", label="sample points", color="Red")
                axs[0].plot(risks, returns, label="efficient frontier", color="Red")
                axs[0].set_xlabel("Standard deviation")
                axs[0].set_ylabel("Expected return")
                axs[0].legend()
                axs[0].grid()

                # Axis 1: The number of open positions
                axs[1].scatter(x=risks, y=npos, color="Red")
                axs[1].plot(risks, npos, color="Red")
                axs[1].set_xlabel("Standard deviation")
                axs[1].set_ylabel("Number of positions")
                axs[1].grid()

                plt.show()
            elif model.status == GRB.INFEASIBLE:
                print("Model is infeasible.")
            elif model.status == GRB.UNBOUNDED:
                print("Model is unbounded.")

    def _setup_constraints(self, model: gp.Model, x: gp.MVar, b: gp.MVar, sigma: pd.DataFrame):
        l = 0.01  # Minimal position size
        V = 3  # Maximal admissible variance (sigma^2)

        # Budget constraint: all investments sum up to 1
        model.addConstr(x.sum() == 1, name="Budget_Constraint")

        # Limit on variance
        risk_constr = model.addConstr(x @ sigma.to_numpy() @ x <= V, name="Variance")

        M = 1.0
        model.addConstr(x <= M * b, name="Upper_Bound_Trade")
        model.addConstr(x >= l * b, name="Lower_Bound_Trade")

        model.addConstr(b.sum() == 4, name="Cardinality")
        return risk_constr
            
    def _apply_black_litterman(self,
                               sigma: pd.DataFrame) -> pd.DataFrame:
        current_weights = self.market_caps / self.market_caps.sum() \
            if self.market_caps is not None else np.ones(self.num_assets) / self.num_assets
        bl = BlackLittermanSignal(current_weights, sigma, self.prices)
        return pd.DataFrame(bl.mean_returns(), index=self.model_data["Mean"].index, columns=["BL_Mean"]).squeeze()
        
class TrackingErrorOptimizer(Optimizer, opt_type = "TrackingErrorOptimizer"):
    def __init__(self,
                 request: dict):
        super().__init__()
        self.request = request
        self.model_data = self.request.get("model_data", [])
        self.num_assets = len(self.model_data["Mean"].head(50))
        random_weights = np.random.rand(self.num_assets)
        self.benchmark_weights = random_weights / np.sum(random_weights)

    def run_optimizer(self):
        # Sample Data
        mu = self.model_data["Mean"].head(self.num_assets)
        Sigma = self.model_data["CovMatrix"].head(self.num_assets)

        # Values for the model parameters:
        r = 0.25  # Required return
        l = 0.00001  # Minimal position size
        u = 0.25 # Max position size
        K = 15 # Max Assets

        # Setup model
        m = gp.Model("TrackingErrorOptimizer")

        # Add variables: x[i] denotes the proportion invested in stock i
        x = m.addMVar(len(mu), lb=0, ub=1, name="x")

        # Add variables: b[i]=1 if stock i is held, and b[i]=0 otherwise
        b = m.addMVar(len(mu), vtype=gp.GRB.BINARY, name="b")

        # Budget constraint: all investments sum up to 1
        m.addConstr(x.sum() == 1, "Budget_Constraint")

        # Lower bound on expected return
        m.addConstr(mu.to_numpy() @ x >= r, "Minimal_Return")

        # Force x to 0 if not traded; see formula (1) above
        m.addConstr(x <= b, name="Indicator")

        # Minimal position; see formula (2) above
        m.addConstr(x >= l * b, name="Minimal_Position")

        m.addConstr(x <= u * b, name="Maximal_Position")

        # Cardinality constraint: at most K positions
        cardinality_constr = m.addConstr(b.sum() <= K, "Cardinality")

        # Create tracking error objective
        weight_diff = x - self.benchmark_weights
        tracking_error_variance = weight_diff @ Sigma.values[:, :self.num_assets] @ weight_diff 

        m.setObjective(tracking_error_variance, GRB.MINIMIZE)

        m.optimize()

        print(f"Minimum Risk:     {m.ObjVal:.6f}")
        print(f"Expected return:  {mu @ x.X:.6f}")
        print(f"Solution time:    {m.Runtime:.2f} seconds\n")
        print(f"Number of trades: {sum(b.X)}\n")

        # Print investments (with non-negligible value, i.e. >1e-5)
        positions = pd.Series(name="Position", data=x.X, index=mu.index)
        print(positions[positions > 1e-5])

class TransactionCostOptimizer(Optimizer, opt_type = "TransactionCostOptimizer"):
    def __init__(self,
                 request: dict):
        super().__init__()
        
        self.request = request
        self.model_data = self.request.get("model_data", [])        

    def run_optimizer(self):
        """Define and solve the optimization model."""
        with gp.Model("TransactionCostOptimizer") as model:
            mu = self.model_data["Mean"]
            x0 = pd.Series(index=mu.index, data=np.zeros(mu.shape))
            x0.loc[mu.nlargest(20).index] = 1.0 / 20.0

            model_params = self._define_model_parameters(mu)

            decision_variables = self._setup_decision_variables(model, mu)

            budget_constr = self._setup_constraints(model, model_params, decision_variables, x0)

            self._setup_objectives(model, decision_variables)

            model.Params.MIPGap = 0.01
            model.optimize()

            results = self._output_results(model, model_params, decision_variables, x0)
            return results
        
    def _define_model_parameters(self, mu: pd.DataFrame):
        # Values for the model parameters:
        return { 
            "V": 4.0,  # Maximal admissible variance (sigma^2)
            "l": 0.001,  # Minimal transaction size

            # Fixed transaction costs
            "c_plus": 0.00002 * np.ones(mu.shape),
            "c_minus": 0.00002 * np.ones(mu.shape),
            # Variable transaction fees
            "f_plus": 0.0002 * np.ones(mu.shape),
            "f_minus": 0.0002 * np.ones(mu.shape) 
        }

    def _setup_decision_variables(self, model: gp.Model, mu: pd.DataFrame):
        decision_variables = DecisionVariables(model, mu)
        return decision_variables

    def _setup_constraints(self, 
                           model: gp.Model, 
                           model_params: dict, 
                           decision_variables: DecisionVariables, 
                           x0: pd.Series):
        mu = self.model_data["Mean"]
        Sigma = self.model_data["CovMatrix"]

        variance = model_params.get("variance", 4.0)
        l = model_params.get("l", 0.001)
        c_plus = model_params.get("c_plus", 0.00002 * np.ones(mu.shape))
        c_minus = model_params.get("c_minus", 0.00002 * np.ones(mu.shape))
        f_plus = model_params.get("f_plus", 0.0002 * np.ones(mu.shape))
        f_minus = model_params.get("f_minus", 0.0002 * np.ones(mu.shape) )

        x_plus = decision_variables.x_plus
        x_minus = decision_variables.x_minus
        b_plus = decision_variables.b_plus
        b_minus = decision_variables.b_minus
        x = decision_variables.x

        model.addConstr(x == x0.to_numpy() + x_plus - x_minus, name="Position_Balance")
        model.addConstr(x @ Sigma.to_numpy() @ x <= variance, name="Variance")

        # Force x_plus, x_minus to 0 if not traded; see formula (2) above
        model.addConstr(x_plus <= b_plus, name="Indicator_Buy")
        model.addConstr(x_minus <= b_minus, name="Indicator_Sell")
        # Minimal buy/sell; see formula (3) above
        model.addConstr(x_plus >= l * b_plus, name="Minimal_Buy")
        model.addConstr(x_minus >= l * b_minus, name="Minimal_Sell")

        # Constraint preventing the simultaneous buy and sell of the same security
        model.addConstr(b_plus + b_minus <= 1, name="Mutual_Exclusivity")
        budget_constr = model.addConstr(
            x.sum() + c_plus @ b_plus + c_minus @ b_minus + f_plus @ x_plus + f_minus @ x_minus
            == 1,
            name="Budget_Constraint",
        )
        return budget_constr

    def _setup_objectives(self, m: gp.Model, decision_variables: DecisionVariables):
        """Define objective function: Maximize expected return minus transaction costs
        """
        mu = self.model_data["Mean"]
        x = decision_variables.x
        m.setObjective(mu.to_numpy() @ x, gp.GRB.MAXIMIZE)

    def _output_results(self, 
                        model: gp.Model, 
                        model_params: dict, 
                        decision_variables: DecisionVariables, 
                        x0: pd.Series):
        mu = self.model_data["Mean"]
        Sigma = self.model_data["CovMatrix"]
        c_plus = model_params.get("c_plus", 0.00002 * np.ones(mu.shape))
        c_minus = model_params.get("c_minus", 0.00002 * np.ones(mu.shape))
        f_plus = model_params.get("f_plus", 0.0002 * np.ones(mu.shape))
        f_minus = model_params.get("f_minus", 0.0002 * np.ones(mu.shape) )
        x_plus = decision_variables.x_plus
        x_minus = decision_variables.x_minus
        b_plus = decision_variables.b_plus
        b_minus = decision_variables.b_minus
        x = decision_variables.x

        print(f"Expected return:  {model.ObjVal:.6f}")
        print(f"Variance:         {x.X @ Sigma @ x.X:.6f}")
        print(f"Solution time:    {model.Runtime:.2f} seconds\n")
        print(f"Fixed costs:      {c_plus @ b_plus.X + c_minus @ b_minus.X:.6f}")
        print(f"Variable fees:    {f_plus @ x_plus.X + f_minus @ x_minus.X:.6f}\n")
        print(
            f"Number of positions: before {np.count_nonzero(x0[x0>1e-5])}, after {np.count_nonzero(x.X[x.X>1e-5])}"
        )
        print(
            f"Number of trades:    {sum(b_plus.X.round()) + sum(b_minus.X.round())} ({sum(b_plus.X.round())} buy(s), {sum(b_minus.X.round())} sell(s))\n"
        )

        # Print all assets with either a non-negligible position or transaction
        results = pd.DataFrame(
            index=mu.index,
            data={
                "position": x.X,
                "transaction": x_plus.X - x_minus.X,
            },
        ).round(6)
        results[(results["position"] > 1e-5) | (abs(results["transaction"]) > 1e-5)].sort_values(
            "position", ascending=False
        )

        return results

class QPOptimizer:
    def __init__(self,
                 config: dict):
        self.config = config
        self.n_assets = config.get("n_assets", 10)
        self.historical_returns = config.get("historical_returns", np.ndarray((100, 10)))
        self.expected_returns = self.historical_returns.mean(axis=0)
        self.tau = config.get("tau", 10)
        self.prices = np.ceil(20 * np.random.rand(self.n_assets))
        raw_cov_matrix = np.cov(self.historical_returns, rowvar=False)
        self.cov = 0.5 * (raw_cov_matrix + raw_cov_matrix.T)

    def optimize(self):
        decision_variables = self._setup_decision_variables()
        constraints = self._setup_constraints(decision_variables)
        objective = self._setup_objective(decision_variables)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver = cp.GUROBI)

        if prob.status == cp.OPTIMAL:
            portfolio_amt = decision_variables.get("portfolio_amt")
            print("Optimal Value = " + str(prob.value))
            print("Optimal Portfolio = " + str(portfolio_amt.value * 100))
            print("Shares = " + str(portfolio_amt.value * 100 / self.prices))
            y = self.prices * np.floor(portfolio_amt.value * 100 / self.prices)
            print("Implemented Portfolio = " + str(y))
            print("Objective Value with implemented portfolio = " + str(portfolio_amt.value @ self.expected_returns - self.tau * y @ self.cov @ y))

    def _setup_decision_variables(self):
        return {
            "portfolio_amt": cp.Variable(self.n_assets, nonneg=True)
        }

    def _setup_constraints(self, 
                           decision_variables: dict):
        portfolio_amt = decision_variables.get("portfolio_amt")
        return [
            cp.sum(portfolio_amt) <= 10000
        ]
    
    def _setup_objective(self, 
                         decision_variables: dict):
        portfolio_amt = decision_variables.get("portfolio_amt", None)
        return_obj = portfolio_amt @ self.expected_returns
        L = np.linalg.cholesky(self.cov)
        risk_obj = cp.quad_form(portfolio_amt, cp.psd_wrap(L)) 
        return cp.Maximize(return_obj - self.tau * risk_obj)

def build_weights(signal_by_date):
    def _w(x):
        r = x.rank() - (len(x) + 1) / 2
        return r / r.abs().sum()
    return signal_by_date.groupby(level="SignalDates").transform(_w)


if __name__ == '__main__':
    # config = {
    #     "n_assets": 10,
    #     "historical_returns": 5 * np.random.rand(100, 10),
    #     "tau": 0.2
    # }
    # optimizer = QPOptimizer(config)
    # optimizer.optimize()

    rng = np.random.default_rng()
    signals_wide = pd.DataFrame({
        "SignalDates": ["2016-01-01", "2017-01-01", "2018-01-01", 
                        "2019-01-01", "2020-01-01", "2021-01-01", 
                        "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
        "Asset_1": rng.random(10),
        "Asset_2": rng.random(10),
        "Asset_3": rng.random(10),
        "Asset_4": rng.random(10),
        "Asset_5": rng.random(10),
        "Asset_6": rng.random(10),
        "Asset_7": rng.random(10),
        "Asset_8": rng.random(10),
        "Asset_9": rng.random(10),
        "Asset_10": rng.random(10)
    })

    signals_by_date = (
        signals_wide
        .set_index("SignalDates")
        .stack()
    )
    signals_by_date.index = signals_by_date.index.set_names(["SignalDates", "Asset"])
    weights = build_weights(signals_by_date)
    print(weights)