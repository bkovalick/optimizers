import numpy as np
from highspy import Highs
import highspy

# 1. Data Setup
n_assets = 10
expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.05, 0.20, 0.10, 0.03, 0.09, 0.10])
# Minimum and maximum allowable weights if an asset is held
min_weight = 0.05 
max_weight = 0.5

# 2. Initialize Model
h = Highs()

# 3. Decision Variables
# w[i]: weight of asset i (continuous, 0 <= w[i] <= 1)
# y[i]: binary variable (1 if asset i is held, 0 otherwise)
w = [h.addVariable(lb=0.0, ub=1.0) for _ in range(n_assets)]
y = [h.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger) for _ in range(n_assets)]

# 4. Objective: Maximize Return (equivalent to Min -Return)
h.changeObjectiveSense(highspy.ObjSense.kMaximize)
for i in range(n_assets):
    h.changeColCost(i, expected_returns[i])

# 5. Constraints
# A. Sum of weights = 1
h.addConstr(sum(w) == 1.0)

# B. Cardinality Constraint: Limit total number of assets (e.g., max 3)
h.addConstr(sum(y) >= 5)

# C. Linking constraint: min_weight*y[i] <= w[i] <= max_weight*y[i]
for i in range(n_assets):
#     h.addConstr(w[i] <= max_weight * y[i])
    h.addConstr(w[i] >= min_weight * y[i])

# 6. Solve
h.run()

# 7. Output Results
if highspy.SolutionStatus.kSolutionStatusFeasible:
    weights = h.getSolution().col_value[:n_assets]
    selected = h.getSolution().col_value[n_assets:]
    print("Optimal Weights:", weights)
    print("Assets Selected (Binary):", selected)
    print("Expected Portfolio Return:", h.getObjectiveValue())
else:
    print("Solution not found.")