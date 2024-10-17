from jormungandr.optimization import OptimizationProblem
from jormungandr.autodiff import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import math
import arm_dynamics

# Parameters
TARGET_ANGLE = 0
TOTAL_SECONDS = 5.0  
TIME_STEP = 0.05  
STEP_COUNT = int(TOTAL_SECONDS / TIME_STEP)

problem = OptimizationProblem()

states = problem.decision_variable(STEP_COUNT+1, 2) # Matrix: (angle, rotational velocity) x simulation steps
efforts = problem.decision_variable(STEP_COUNT, 1) # Matrix: (torque effort) x simulation steps
angles = states[:, 0]
vels = states[:, 1]

# Constraints
problem.subject_to(angles[0] == -2*np.pi) # initial angle (pointed down)
problem.subject_to(vels[0] == 0.0) # initial velocity (not moving)

for k in range(STEP_COUNT):
    next_angle, next_vel = arm_dynamics.next_state(angles[k], vels[k], efforts[k], TIME_STEP)
    problem.subject_to(angles[k+1] == next_angle)
    problem.subject_to(vels[k+1] == next_vel)

problem.subject_to(efforts <= 20.0)
problem.subject_to(efforts >= -20.0)
problem.subject_to(angles[-1] == TARGET_ANGLE)
problem.subject_to(vels[-1] == 0.0)

# Objective function (minimize distance from target angle)
cost = 0.0
for k in range(STEP_COUNT+1):
    cost += (TARGET_ANGLE - angles[k])**2
problem.minimize(cost)

# Solve
problem.solve()

# Extract the solution
solved_angles = [p.value() for p in angles]
solved_velocities = [v.value() for v in vels]
solved_efforts = [e.value() for e in efforts]

# Plot the results
time = np.linspace(0, TOTAL_SECONDS, STEP_COUNT+1)
target_angles = [TARGET_ANGLE] * (STEP_COUNT+1)
plt.figure()
plt.plot(time, solved_angles, label='angle (rad)')
plt.plot(time, target_angles, label='target angle (rad)')
plt.plot(time[:-1], solved_efforts, label='effort (N)')
plt.xlabel('Time [s]')
plt.ylabel('States / Control')
plt.legend(loc='upper right')
plt.show()