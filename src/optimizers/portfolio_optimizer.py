"""
Portfolio Optimization with Minimum Trade Size and Liquidation Constraints

This module implements a portfolio optimization framework that:
- Decomposes trades into Cash and Turnover components
- Enforces minimum trade size constraints
- Allows liquidation even if it violates minimum trade size
- Minimizes deviation from target weights
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


class PortfolioOptimizer:
    """
    Portfolio optimizer with minimum trade size and liquidation constraints.
    
    Trades are decomposed into:
    - Cash: Allocation of net cash flow
    - Turnover: Cash-neutral rebalancing trades
    
    Total trade = Cash + Turnover must satisfy:
    - |Total trade| >= MinTradeSize OR
    - Total trade = 0 (no trade) OR
    - Total trade = -holdings (full liquidation)
    """
    
    def __init__(self, env):
        """Initialize optimizer with Gurobi environment."""
        self.env = env
        self.model = None
        
        # Data attributes
        self.assets = None
        self.holdings = None
        self.target_weights = None
        self.net_cash = None
        self.target_turnover = None
        self.min_trade_size = None
        
        # Variables
        self.cash = None
        self.turnover = None
        self.total_trade = None
        self.new_weights = None
        self.is_trading = None  # Binary: trade or not
        self.liquidation = None  # Binary: liquidate or not
        self.abs_turnover = None
        
    def __enter__(self):
        """Enter context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False
    
    def set_data(self, holdings, target_weights, net_cash, target_turnover, min_trade_size):
        """
        Set portfolio data.
        
        Parameters
        ----------
        holdings : dict
            Current holdings in dollars {asset: amount}
        target_weights : dict
            Target portfolio weights {asset: weight}
        net_cash : float
            Net cash flow to allocate (positive for contribution, negative for withdrawal)
        target_turnover : float
            Maximum allowable turnover in dollars
        min_trade_size : float
            Minimum trade size in dollars
        """
        self.assets = list(holdings.keys())
        self.holdings = holdings
        self.target_weights = target_weights
        self.net_cash = net_cash
        self.target_turnover = target_turnover
        self.min_trade_size = min_trade_size
        
        # Validate data
        assert set(holdings.keys()) == set(target_weights.keys()), \
            "Holdings and target weights must have the same assets"
        assert abs(sum(target_weights.values()) - 1.0) < 1e-6, \
            "Target weights must sum to 1.0"
    
    def build_model(self):
        """Build the optimization model."""
        self.model = gp.Model("PortfolioOptimization", env=self.env)
        self._create_variables()
        self._add_trade_definition()
        self._add_cash_turnover_constraints()
        self._add_minimum_trade_size_constraints()
        self._add_weight_constraints()
        self._set_objective()
        
    def _create_variables(self):
        """Create decision variables."""
        # Cash and turnover variables (unrestricted)
        self.cash = self.model.addVars(
            self.assets, 
            lb=-GRB.INFINITY, 
            name="Cash"
        )
        self.turnover = self.model.addVars(
            self.assets, 
            lb=-GRB.INFINITY, 
            name="Turnover"
        )
        
        # Total trade (unrestricted)
        self.total_trade = self.model.addVars(
            self.assets, 
            lb=-GRB.INFINITY, 
            name="TotalTrade"
        )
        
        # Binary variables for trade decisions
        self.is_trading = self.model.addVars(
            self.assets, 
            vtype=GRB.BINARY, 
            name="Trade"
        )

        self.liquidation = self.model.addVars(
            self.assets, 
            vtype=GRB.BINARY, 
            name="Liquidate"
        )

        self.is_buying = self.model.addVars(
            self.assets, 
            vtype=GRB.BINARY, 
            name=f"IsBuying"
        )
        
        self.is_selling = self.model.addVars(
            self.assets, 
            vtype=GRB.BINARY, 
            name=f"IsSelling"
        )
        
        self.no_trading = self.model.addVars(
            self.assets, 
            vtype=GRB.BINARY, 
            name=f"NoTrading"
        )
        
        # Absolute value of turnover for budget constraint
        self.abs_turnover = self.model.addVars(
            self.assets, 
            lb=0, 
            name="AbsTurnover"
        )
        
        # New weights (non-negative)
        self.new_weights = self.model.addVars(
            self.assets, 
            lb=0, 
            ub=1, 
            name="NewWeight"
        )
        
    def _add_trade_definition(self):
        """Define total trade as Cash + Turnover."""
        self.model.addConstrs(
            (self.total_trade[i] == self.cash[i] + self.turnover[i] 
             for i in self.assets),
            name="TotalTradeDef"
        )
        
    def _add_cash_turnover_constraints(self):
        """Add cash and turnover budget constraints."""
        # Cash constraint: sum of cash allocations equals net cash flow
        self.model.addConstr(
            gp.quicksum(self.cash[i] for i in self.assets) == self.net_cash,
            name="CashBudget"
        )
        
        # Turnover is cash-neutral
        self.model.addConstr(
            gp.quicksum(self.turnover[i] for i in self.assets) == 0,
            name="TurnoverNeutral"
        )
        
        # Absolute value of turnover
        for i in self.assets:
            self.model.addGenConstrAbs(
                self.abs_turnover[i], 
                self.turnover[i], 
                name=f"AbsTurnover_{i}"
            )
        
        # Turnover budget constraint
        self.model.addConstr(
            gp.quicksum(self.abs_turnover[i] for i in self.assets) <= self.target_turnover,
            name="TurnoverBudget"
        )
        
    def _add_minimum_trade_size_constraints(self):
        """Add minimum trade size constraints with liquidation exception."""
        for i in self.assets:
            h_i = self.holdings[i]

            # Only one binary variable can be active at a time.
            self.model.addSOS(GRB.SOS_TYPE1, 
                              [self.no_trading[i], self.liquidation[i], self.is_buying[i], self.is_selling[i]], 
                              [1, 2, 3, 4])
            
            self.model.addConstr(self.no_trading[i] + self.liquidation[i] + self.is_buying[i] + self.is_selling[i] == 1)

            # Constraint 1: If liquidate, then total_trade = -holdings
            self.model.addGenConstrIndicator(
                self.liquidation[i], 
                1, 
                self.total_trade[i], 
                GRB.EQUAL, 
                -h_i,
                name=f"Liquidation_{i}"
            )
            
            # Constraint 2: If no trade, then total_trade = 0
            self.model.addGenConstrIndicator(
                self.no_trading[i], 
                1, 
                self.total_trade[i], 
                GRB.EQUAL, 
                0,
                name=f"NoTrade_{i}"
            )

            
            # Constraint 3: If buy, then total_trade >= MinTradeSize
            self.model.addGenConstrIndicator(
                self.is_buying[i],
                1,
                self.total_trade[i],
                GRB.GREATER_EQUAL,
                self.min_trade_size,
                name=f"MinBuy_{i}"
            )
            
            # Constraint 4: If sell, then total_trade <= -MinTradeSize
            self.model.addGenConstrIndicator(
                self.is_selling[i],
                1,
                self.total_trade[i],
                GRB.LESS_EQUAL,
                -self.min_trade_size,
                name=f"MinSell_{i}"
            )
    
    def _add_weight_constraints(self):
        """Define new portfolio weights."""
        # Calculate new portfolio value
        current_value = sum(self.holdings.values())
        new_value = current_value + self.net_cash
        
        # Define new weights: w_i = (h_i + total_trade_i) / new_value
        for i in self.assets:
            self.model.addConstr(
                self.new_weights[i] * new_value == self.holdings[i] + self.total_trade[i],
                name=f"WeightDef_{i}"
            )
        
        # Weights must sum to 1
        self.model.addConstr(
            gp.quicksum(self.new_weights[i] for i in self.assets) == 1.0,
            name="WeightSum"
        )
    
    def _set_objective(self):
        """Set objective: minimize squared deviation from target weights."""
        # Quadratic objective: sum of (new_weight - target_weight)^2
        obj = gp.quicksum(
            (self.new_weights[i] - self.target_weights[i]) ** 2 
            for i in self.assets
        )
        self.model.setObjective(obj, GRB.MINIMIZE)
    
    def solve(self):
        """Solve the optimization model."""
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            print(f"\nOptimal solution found!")
            print(f"Objective value: {self.model.ObjVal:.6f}")
            return GRB.OPTIMAL
        elif self.model.Status == GRB.INFEASIBLE:
            print("\nModel is infeasible.")
            print("Computing IIS...")
            self.model.computeIIS()
            self.model.write("portfolio_infeasible.ilp")
            print("IIS written to portfolio_infeasible.ilp")
            return GRB.INFEASIBLE
        else:
            print(f"\nOptimization ended with status {self.model.Status}")
            return self.model.Status
    
    def get_solution(self):
        """Extract solution as a pandas DataFrame."""
        if self.model.Status != GRB.OPTIMAL:
            return None
        
        solution = []
        for i in self.assets:
            solution.append({
                'Asset': i,
                'CurrentHoldings': self.holdings[i],
                'Cash': self.cash[i].X,
                'Turnover': self.turnover[i].X,
                'TotalTrade': self.total_trade[i].X,
                'NewHoldings': self.holdings[i] + self.total_trade[i].X,
                'TargetWeight': self.target_weights[i],
                'NewWeight': self.new_weights[i].X,
                'Trade': self.is_trading[i].X,
                'Liquidate': self.liquidation[i].X
            })
        
        return pd.DataFrame(solution)
    
    def display_results(self):
        """Display optimization results."""
        df = self.get_solution()
        if df is None:
            print("No solution available.")
            return
        
        print("\n" + "="*100)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("="*100)
        
        # Summary statistics
        current_value = sum(self.holdings.values())
        new_value = current_value + self.net_cash
        total_turnover = df['Turnover'].abs().sum()
        
        print(f"\nPortfolio Summary:")
        print(f"  Current Value:    ${current_value:,.2f}")
        print(f"  Net Cash Flow:    ${self.net_cash:,.2f}")
        print(f"  New Value:        ${new_value:,.2f}")
        print(f"  Total Turnover:   ${total_turnover:,.2f} (Limit: ${self.target_turnover:,.2f})")
        print(f"  Min Trade Size:   ${self.min_trade_size:,.2f}")
        
        # Display trades
        trades_df = df[df['TotalTrade'].abs() > 1e-6].copy()
        if len(trades_df) > 0:
            print(f"\nTrades ({len(trades_df)} assets):")
            print("-" * 100)
            for _, row in trades_df.iterrows():
                trade_type = "LIQUIDATE" if row['Liquidate'] > 0.5 else ("BUY" if row['TotalTrade'] > 0 else "SELL")
                print(f"  {row['Asset']:10s} {trade_type:10s}  "
                      f"Cash: ${row['Cash']:10,.2f}  "
                      f"Turnover: ${row['Turnover']:10,.2f}  "
                      f"Total: ${row['TotalTrade']:10,.2f}")
        else:
            print("\nNo trades executed.")
        
        # Display weights
        print(f"\nWeight Comparison:")
        print("-" * 100)
        print(f"{'Asset':10s} {'Target':>10s} {'New':>10s} {'Deviation':>10s}")
        print("-" * 100)
        for _, row in df.iterrows():
            deviation = row['NewWeight'] - row['TargetWeight']
            print(f"{row['Asset']:10s} {row['TargetWeight']:10.4f} "
                  f"{row['NewWeight']:10.4f} {deviation:10.4f}")
        
        print("="*100)


def main():
    """Main function demonstrating the portfolio optimizer."""
    
    # Sample data: 5-asset portfolio
    holdings = {
        'AAPL': 50000,
        'GOOGL': 30000,
        'MSFT': 40000,
        'AMZN': 25000,
        'TSLA': 15000
    }
    
    # Target weights (sum to 1.0)
    target_weights = {
        'AAPL': 0.35,
        'GOOGL': 0.25,
        'MSFT': 0.20,
        'AMZN': 0.20,
        'TSLA': 0.0
    }
    
    # Portfolio parameters
    net_cash = 20000  # Contributing $20,000
    target_turnover = 30000  # Maximum $30,000 in turnover
    min_trade_size = 5000  # Minimum trade size of $5,000
    
    # Create Gurobi environment
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1)
        env.start()
        
        # Use the optimizer with context manager
        with PortfolioOptimizer(env) as optimizer:
            optimizer.set_data(
                holdings=holdings,
                target_weights=target_weights,
                net_cash=net_cash,
                target_turnover=target_turnover,
                min_trade_size=min_trade_size
            )
            optimizer.build_model()
            
            status = optimizer.solve()
            
            if status == GRB.OPTIMAL:
                optimizer.display_results()


if __name__ == "__main__":
    main()
