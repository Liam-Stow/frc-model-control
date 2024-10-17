import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import arm_dynamics

# Parameters
TARGET_ANGLE = 0.0
TIME_STEP = 0.05
TOTAL_SECONDS = 5.0
STEP_COUNT = int(TOTAL_SECONDS / TIME_STEP)

# Decision variables for state trajectory and control inputs
opti = ca.Opti()
states = opti.variable(2, STEP_COUNT+1)
angles = states[0, :]
velocities = states[1, :]
control_torques = opti.variable(1, STEP_COUNT)

# Constraints
opti.subject_to(angles[0] == -2*np.pi)  # initial angle (pointed down)
opti.subject_to(velocities[0] == 0)  # initial velocity (not moving)

for k in range(STEP_COUNT):
    next_angle, next_vel = arm_dynamics.next_state(angles[k], velocities[k], control_torques[k], TIME_STEP)
    opti.subject_to(angles[k+1] == next_angle)
    opti.subject_to(velocities[k+1] == next_vel)

opti.subject_to(opti.bounded(-20, control_torques, 20)) 
opti.subject_to(angles[-1] == TARGET_ANGLE)  # final angle = target
opti.subject_to(velocities[-1] == 0)  # final velocity = 0

# Objective function (minimize distance from target angle)
opti.minimize(ca.sumsqr(angles - TARGET_ANGLE))

# Solve
opti.solver('ipopt')
solution = opti.solve()

# Extract the solution
solved_angles = solution.value(angles)
solved_velocities = solution.value(velocities)
solved_control_torques = solution.value(control_torques)

# Plot the results
time = np.linspace(0, TOTAL_SECONDS, STEP_COUNT+1)
target_angles = [TARGET_ANGLE] * (STEP_COUNT+1)
plt.figure()
plt.plot(time, solved_angles, label='angle (rad)')
plt.plot(time, target_angles, label='target angle (rad)')
plt.plot(time[:-1], solved_control_torques, label='control (N)')
plt.xlabel('Time [s]')
plt.ylabel('States / Control')
plt.legend(loc='upper right')
plt.show()
