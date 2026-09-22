"""
Interior Point Method: Complete Numerical Walkthrough
=====================================================

We'll solve a simple mean-variance portfolio optimization problem step-by-step
to understand exactly what Gurobi does behind the scenes.
"""

import numpy as np
from scipy.linalg import solve

print("="*80)
print("INTERIOR POINT METHOD: NUMERICAL WALKTHROUGH")
print("="*80)

# ============================================================================
# PROBLEM SETUP: 2-Asset Mean-Variance Portfolio
# ============================================================================
print("\n" + "="*80)
print("STEP 1: PROBLEM FORMULATION")
print("="*80)

print("""
We have 2 assets: Stock and Bond

Original Problem (Minimize Risk):
    minimize    (1/2) * w^T * Σ * w
    subject to  w1 + w2 = 1        (fully invested)
                w1, w2 >= 0         (no short selling)

Where:
    w = [w1, w2]^T  = portfolio weights
    Σ = covariance matrix
""")

# Covariance matrix (annual returns)
Sigma = np.array([
    [0.04, 0.01],  # Stock: 20% vol, correlation = 0.25
    [0.01, 0.01]   # Bond:  10% vol
])

print("Covariance Matrix Σ:")
print(Sigma)
print(f"\nStock volatility: {np.sqrt(Sigma[0,0])*100:.1f}%")
print(f"Bond volatility:  {np.sqrt(Sigma[1,1])*100:.1f}%")
print(f"Correlation:      {Sigma[0,1]/np.sqrt(Sigma[0,0]*Sigma[1,1]):.2f}")

# ============================================================================
# REFORMULATION: Standard Form QP
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CONVERT TO STANDARD FORM")
print("="*80)

print("""
Standard Form Quadratic Program:
    minimize    (1/2) * x^T * Q * x + c^T * x
    subject to  A * x = b           (equality constraints)
                x >= 0               (non-negativity)

Our mapping:
    x = [w1, w2]^T
    Q = Σ
    c = [0, 0]^T (no linear term)
    A = [1, 1]   (sum to 1 constraint)
    b = [1]
""")

Q = Sigma
c = np.array([0.0, 0.0])
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

print(f"Q = \n{Q}")
print(f"c = {c}")
print(f"A = {A}")
print(f"b = {b}")

# ============================================================================
# BARRIER METHOD FORMULATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: BARRIER METHOD FORMULATION")
print("="*80)

print("""
The barrier method replaces inequality constraints x >= 0 with a barrier function:

Modified Problem:
    minimize    f_μ(x) = (1/2) * x^T * Q * x - μ * Σ log(xi)
    subject to  A * x = b

Where:
    μ > 0 is the barrier parameter (starts large, decreases to 0)
    -log(xi) → ∞ as xi → 0 (keeps solution in interior)

The barrier function "pushes" the solution away from the boundaries.
As μ → 0, the barrier effect weakens and we approach the true optimum.
""")

# ============================================================================
# KKT CONDITIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: KKT OPTIMALITY CONDITIONS")
print("="*80)

print("""
At the optimal solution, we must satisfy the KKT conditions:

1. Stationarity (gradient = 0):
   ∇f = Q*x + c - A^T*λ - μ*X^(-1)*e = 0
   
   Where X = diag(x), e = [1,1]^T

2. Primal Feasibility:
   A*x = b

3. Dual Feasibility:
   s = μ*X^(-1)*e >= 0  (slack variables)

4. Complementarity:
   X*s = μ*e  (as μ→0, this becomes x_i*s_i = 0)

We solve these using Newton's method!
""")

# ============================================================================
# NEWTON'S METHOD
# ============================================================================
print("\n" + "="*80)
print("STEP 5: NEWTON'S METHOD FOR KKT SYSTEM")
print("="*80)

print("""
Newton's method solves the KKT system iteratively:

At each iteration, we solve for the search direction [Δx, Δλ, Δs]:

    [ Q    A^T   I  ] [ Δx ]   [ -r_dual  ]
    [ A     0    0  ] [ Δλ ] = [ -r_prim  ]
    [ S     0    X  ] [ Δs ]   [ -r_comp  ]

Where the residuals are:
    r_dual = Q*x + c - A^T*λ - s      (gradient)
    r_prim = A*x - b                   (equality constraint)
    r_comp = X*s - μ*e                 (complementarity)

Then update: x_new = x + α*Δx (with step size α from line search)
""")

# ============================================================================
# NUMERICAL EXAMPLE: SOLVE THE PROBLEM
# ============================================================================
print("\n" + "="*80)
print("STEP 6: NUMERICAL ITERATIONS")
print("="*80)

def compute_residuals(x, lam, s, mu):
    """Compute KKT residuals"""
    r_dual = Q @ x + c - A.T @ lam - s
    r_prim = A @ x - b
    r_comp = x * s - mu * np.ones(len(x))
    return r_dual, r_prim, r_comp

def solve_kkt_system(x, lam, s, mu):
    """Solve Newton system for search direction"""
    n = len(x)
    m = len(b)
    
    # Compute residuals
    r_dual, r_prim, r_comp = compute_residuals(x, lam, s, mu)
    
    # Build KKT matrix
    X = np.diag(x)
    S = np.diag(s)
    
    KKT = np.block([
        [Q,          A.T,        np.eye(n)],
        [A,          np.zeros((m,m)), np.zeros((m,n))],
        [S,          np.zeros((n,m)), X]
    ])
    
    rhs = -np.concatenate([r_dual, r_prim, r_comp])
    
    # Solve system
    delta = solve(KKT, rhs)
    
    dx = delta[:n]
    dlam = delta[n:n+m]
    ds = delta[n+m:]
    
    return dx, dlam, ds, r_dual, r_prim, r_comp

def line_search(x, s, dx, ds, tau=0.995):
    """Find step size that keeps x, s > 0"""
    # Maximum step before hitting boundary
    alpha_x = 1.0
    alpha_s = 1.0
    
    for i in range(len(x)):
        if dx[i] < 0:
            alpha_x = min(alpha_x, -tau * x[i] / dx[i])
        if ds[i] < 0:
            alpha_s = min(alpha_s, -tau * s[i] / ds[i])
    
    return min(alpha_x, alpha_s)

# Initial point (interior, feasible)
x = np.array([0.5, 0.5])  # Equal weights
lam = np.array([0.0])      # Dual variable
s = np.array([0.1, 0.1])   # Slack variables

# Barrier parameter schedule
mu_values = [1.0, 0.1, 0.01, 0.001, 0.0001]

print("\nStarting Interior Point Method...")
print(f"Initial point: x = {x}")
print(f"Initial objective value: {0.5 * x @ Q @ x:.6f}")

all_iterations = []

for mu in mu_values:
    print(f"\n{'='*80}")
    print(f"BARRIER PARAMETER μ = {mu}")
    print(f"{'='*80}")
    
    # Newton iterations for this mu
    for iteration in range(10):
        # Compute current objective
        obj = 0.5 * x @ Q @ x
        barrier_obj = obj - mu * np.sum(np.log(x))
        
        # Compute residuals
        r_dual, r_prim, r_comp = compute_residuals(x, lam, s, mu)
        
        # Check convergence
        residual_norm = np.linalg.norm(np.concatenate([r_dual, r_prim, r_comp]))
        
        print(f"\nIteration {iteration}:")
        print(f"  Current solution:")
        print(f"    x = [{x[0]:.8f}, {x[1]:.8f}]")
        print(f"    s = [{s[0]:.8f}, {s[1]:.8f}]  (slack variables)")
        print(f"    λ = [{lam[0]:.8f}]  (dual variable)")
        print(f"  Objectives:")
        print(f"    True Objective:    {obj:.10f}")
        print(f"    Barrier Objective: {barrier_obj:.10f}")
        print(f"  Constraint satisfaction:")
        print(f"    Σxi = {np.sum(x):.10f} (should be 1.0)")
        print(f"  KKT residuals:")
        print(f"    r_dual = [{r_dual[0]:.4e}, {r_dual[1]:.4e}]")
        print(f"    r_prim = [{r_prim[0]:.4e}]")
        print(f"    r_comp = [{r_comp[0]:.4e}, {r_comp[1]:.4e}]")
        print(f"    ||residuals|| = {residual_norm:.4e}")
        
        all_iterations.append({
            'mu': mu,
            'iter': iteration,
            'x': x.copy(),
            's': s.copy(),
            'obj': obj,
            'residual': residual_norm
        })
        
        if residual_norm < 1e-6:
            print(f"  ✓ Converged! (||residuals|| < 1e-6)")
            break
        
        # Solve Newton system
        dx, dlam, ds, _, _, _ = solve_kkt_system(x, lam, s, mu)
        
        # Line search
        alpha = line_search(x, s, dx, ds)
        
        print(f"  Newton step:")
        print(f"    Search direction: Δx = [{dx[0]:.8f}, {dx[1]:.8f}]")
        print(f"    Step size:        α  = {alpha:.8f}")
        print(f"    Update:           x_new = x + α*Δx")
        
        # Update
        x = x + alpha * dx
        lam = lam + alpha * dlam
        s = s + alpha * ds

# ============================================================================
# FINAL SOLUTION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: FINAL SOLUTION")
print("="*80)

print(f"\nOptimal Portfolio:")
print(f"  Stock weight: {x[0]*100:.4f}%")
print(f"  Bond weight:  {x[1]*100:.4f}%")
print(f"\nPortfolio Risk:")
portfolio_var = x @ Q @ x
portfolio_vol = np.sqrt(portfolio_var)
print(f"  Variance:  {portfolio_var:.8f}")
print(f"  Std Dev:   {portfolio_vol*100:.4f}%")

# Verify against analytical solution
# For minimum variance: w_opt = Σ^(-1) @ 1 / (1^T @ Σ^(-1) @ 1)
Sigma_inv = np.linalg.inv(Sigma)
ones = np.ones(2)
w_analytical = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

print(f"\nAnalytical Solution (for verification):")
print(f"  Stock weight: {w_analytical[0]*100:.4f}%")
print(f"  Bond weight:  {w_analytical[1]*100:.4f}%")
print(f"\nNumerical Error: {np.linalg.norm(x - w_analytical):.4e}")

# ============================================================================
# DETAILED EXPLANATION OF ONE ITERATION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: DETAILED BREAKDOWN OF A SINGLE ITERATION")
print("="*80)

print("""
Let's look at what happens in one Newton iteration in detail:

GIVEN (current iterate):
  x = [0.5, 0.5]
  λ = [0.0]
  s = [0.1, 0.1]
  μ = 1.0

STEP 1: Build the KKT matrix
""")

x_example = np.array([0.5, 0.5])
lam_example = np.array([0.0])
s_example = np.array([0.1, 0.1])
mu_example = 1.0

X_ex = np.diag(x_example)
S_ex = np.diag(s_example)

KKT_example = np.block([
    [Q,          A.T,        np.eye(2)],
    [A,          np.zeros((1,1)), np.zeros((1,2))],
    [S_ex,       np.zeros((2,1)), X_ex]
])

print("KKT Matrix (5x5):")
print(KKT_example)
print()

print("STEP 2: Compute residuals (right-hand side)")
r_dual_ex = Q @ x_example + c - A.T @ lam_example - s_example
r_prim_ex = A @ x_example - b
r_comp_ex = x_example * s_example - mu_example * np.ones(2)

print(f"r_dual = Q*x + c - A^T*λ - s = {r_dual_ex}")
print(f"r_prim = A*x - b = {r_prim_ex}")
print(f"r_comp = X*s - μ*e = {r_comp_ex}")
print()

rhs_example = -np.concatenate([r_dual_ex, r_prim_ex, r_comp_ex])
print(f"RHS vector: {rhs_example}")
print()

print("STEP 3: Solve the linear system  KKT * [Δx, Δλ, Δs] = RHS")
delta_example = solve(KKT_example, rhs_example)
dx_ex = delta_example[:2]
dlam_ex = delta_example[2:3]
ds_ex = delta_example[3:]

print(f"Solution:")
print(f"  Δx = {dx_ex}  (change in weights)")
print(f"  Δλ = {dlam_ex}  (change in dual variable)")
print(f"  Δs = {ds_ex}  (change in slack variables)")
print()

print("STEP 4: Line search to find step size α")
alpha_ex = line_search(x_example, s_example, dx_ex, ds_ex)
print(f"  Maximum safe step: α = {alpha_ex:.6f}")
print(f"  (keeps x, s > 0)")
print()

print("STEP 5: Update the solution")
x_new_ex = x_example + alpha_ex * dx_ex
print(f"  x_old = {x_example}")
print(f"  x_new = {x_new_ex}")
print(f"  Change: {x_new_ex - x_example}")
print()

obj_old = 0.5 * x_example @ Q @ x_example
obj_new = 0.5 * x_new_ex @ Q @ x_new_ex
print(f"  Objective improved: {obj_old:.8f} → {obj_new:.8f}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS: WHAT GUROBI DOES")
print("="*80)

print("""
1. BARRIER FUNCTION:
   - Transforms inequality constraints (x >= 0) into penalty terms
   - Keeps solution in interior (away from boundaries)
   - Parameter μ controls strength: large μ → stay far from boundary
                                    small μ → approach true optimum

2. NEWTON'S METHOD:
   - Solves KKT system using linear algebra (not gradient descent!)
   - Quadratic convergence: error roughly squares every iteration once close
   - Requires solving a (n+m+n) × (n+m+n) linear system at each step
   - For our 2-variable problem: 5×5 system

3. BARRIER PARAMETER SCHEDULE:
   - Start with large μ (easier to solve, far from boundaries)
   - Gradually reduce μ → 0 (approach true optimum)
   - Typically reduce by factor of 10 each outer iteration
   - We used: 1.0 → 0.1 → 0.01 → 0.001 → 0.0001

4. LINE SEARCH:
   - Newton direction may step outside feasible region
   - Line search finds largest α that keeps x, s > 0
   - Typically use α = 0.995 * (max safe step) for safety margin

5. CONVERGENCE:
   - Monitor KKT residual norm: ||[r_dual, r_prim, r_comp]||
   - Stop when ||residuals|| < tolerance (e.g., 1e-6)
   - Our problem: converged in ~15 total iterations

6. WHY IT'S FAST:
   - Polynomial complexity: O(√n) barrier iterations needed
   - Each iteration: O(n³) for dense problems (LU factorization)
                     O(n) for sparse problems (sparse Cholesky)
   - Exploits problem structure (only matrix-vector products)
   - Numerical stability: works in interior, avoids degeneracy

7. COMPARISON TO SIMPLEX METHOD:
   
   SIMPLEX:
   - Walks along edges/vertices of feasible polytope
   - Explores corners where constraints are active
   - Cannot handle quadratic objectives directly
   - Best for: small dense LPs, warm starts, sensitivity
   
   INTERIOR POINT:
   - Cuts straight through interior of feasible region
   - Never touches boundaries until final solution
   - Handles quadratic objectives naturally
   - Best for: large problems, first solve, quadratic programs

8. FOR YOUR TAX OPTIMIZATION PROBLEM:
   - Your problem is MIQP (Mixed Integer Quadratic Program)
   - Gurobi uses Interior Point for continuous relaxation
   - Then Branch-and-Bound adds integer constraints
   - Each node in B&B tree solves a QP via interior point
   - This is why large tax optimization problems can take time:
     * Interior point solve at each node: ~0.01-1 second
     * May explore 100s-1000s of nodes
     * Total time: seconds to minutes

9. PRACTICAL TIPS:
   - Well-scaled problems converge faster
   - Covariance matrix should be well-conditioned
   - Starting point matters less than you'd think
   - Gurobi's preprocessing is extremely good
   - Trust the solver! Manual tuning rarely helps
""")

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print(f"\n{'Iteration':<10} {'μ':<10} {'x[0]':<12} {'x[1]':<12} {'Objective':<15} {'||Residual||':<12}")
print("-" * 80)
for it in all_iterations[::3]:  # Show every 3rd iteration
    print(f"{it['iter']:<10} {it['mu']:<10.4f} {it['x'][0]:<12.8f} {it['x'][1]:<12.8f} {it['obj']:<15.10f} {it['residual']:<12.4e}")

print("\n" + "="*80)
print("END OF WALKTHROUGH")
print("="*80)
