import pulp as lp
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate the Synthetic Vital Signs Data
np.random.seed(42)
num_patients = 10
stable_hr = np.random.normal(75, 10, num_patients)
stable_bp = np.random.normal(120, 10, num_patients)
stable_data = list(zip(stable_hr, stable_bp))
at_risk_hr = np.random.normal(110, 15, num_patients)
at_risk_bp = np.random.normal(160, 15, num_patients)
at_risk_data = list(zip(at_risk_hr, at_risk_bp))

# 2. Create the Linear Programming Problem
prob = lp.LpProblem("Medical_Triage_Classification", lp.LpMinimize)

# 3. Define the Decision Variables
w1 = lp.LpVariable("w1", cat="Continuous")
w2 = lp.LpVariable("w2", cat="Continuous")
gamma = lp.LpVariable("gamma", cat="Continuous")
y = [lp.LpVariable(f"y_{i}", lowBound=0) for i in range(num_patients)]
z = [lp.LpVariable(f"z_{i}", lowBound=0) for i in range(num_patients)]

# 4. Define the Objective Function
prob += (lp.lpSum(y) / num_patients) + (lp.lpSum(z) / num_patients)

# 5. Add the Constraints
for i, (hr, bp) in enumerate(stable_data):
    prob += y[i] >= (gamma + 1) - (w1 * hr + w2 * bp)
for i, (hr, bp) in enumerate(at_risk_data):
    prob += z[i] >= (w1 * hr + w2 * bp) - (gamma - 1)

# 6. Solve the Problem
solver = lp.PULP_CBC_CMD(msg=False)
prob.solve(solver)

# 7. Print the Results (CHECK YOUR CONSOLE FOR THIS OUTPUT)
print("Solver Status:", lp.LpStatus[prob.status])
print("Objective Value (Total Avg. Violation):", lp.value(prob.objective))
print("\nOptimal Parameters:")
print(f"w1 = {w1.varValue:.6f}")
print(f"w2 = {w2.varValue:.6f}")
print(f"gamma = {gamma.varValue:.6f}")
print(f"\nSeparating Line: {w1.varValue:.4f}*HR + {w2.varValue:.4f}*BP = {gamma.varValue:.4f}")

# 8. Plot the Results
plt.figure(figsize=(10, 6))
plt.scatter(stable_hr, stable_bp, c='green', marker='o', label='Stable', s=100, edgecolors='black')
plt.scatter(at_risk_hr, at_risk_bp, c='red', marker='x', label='At-Risk', s=100, linewidths=2)
hr_range = np.array([50, 140]) # Use a fixed range for better plot view
bp_range = (gamma.varValue - w1.varValue * hr_range) / w2.varValue
plt.plot(hr_range, bp_range, 'b--', label=f'Separating Line', linewidth=2)
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Medical Triage Classification using LP')
plt.legend()
plt.grid(True)
plt.show(block=True)  # This command keeps the plot window open