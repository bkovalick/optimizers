from itertools import product
import pandas as pd

class RebalanceProblemSweep:
    def __init__(self, build_rebalance_problem):
        self.build_rebalance_problem = build_rebalance_problem

    def build(self, param_sweeps: dict, market_data: pd.DataFrame, signal_type: str = "garch") -> list[dict]:
        configs = []
        for combo in self.combinations(param_sweeps):
            config = {
                **self.build_rebalance_problem(
                    market_data,
                    time_horizon=combo["time_horizon"],
                    signal_type=signal_type,
                ),
                **combo,
            }
            configs.append(config)
        return configs

    def combinations(self, param_sweeps: dict) -> list[dict]:
        keys, values = zip(*param_sweeps.items())
        return [dict(zip(keys, value_set)) for value_set in product(*values)]