import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import arm_dynamics

# Parameters
TARGET_ANGLE = 0.0
TIME_STEP = 0.005
TOTAL_SECONDS = 0.5
STEP_COUNT = int(TOTAL_SECONDS / TIME_STEP)

# Decision variables for state trajectory and control inputs
opti = ca.Opti()
states = opti.variable(2, STEP_COUNT+1)
angles = states[0, :]
velocities = states[1, :]
control_voltages = opti.variable(1, STEP_COUNT)
control_currents = opti.variable(1, STEP_COUNT)

# Constraints
opti.subject_to(angles[0] == -np.pi/2.0)  # initial angle (pointed down)
opti.subject_to(velocities[0] == 0)  # initial velocity (not moving)

for k in range(STEP_COUNT):
    next_angle, next_vel = arm_dynamics.next_state_with_voltage(angles[k], velocities[k], control_voltages[k], TIME_STEP)
    opti.subject_to(angles[k+1] == next_angle)
    opti.subject_to(velocities[k+1] == next_vel)
    opti.subject_to(control_currents[k] == arm_dynamics.calc_current(control_voltages[k], velocities[k]))

opti.subject_to(opti.bounded(-10, control_voltages, 10)) 
opti.subject_to(opti.bounded(-200, control_currents, 200))
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
solved_control_voltages = solution.value(control_voltages)
solved_control_currents = solution.value(control_currents)

# Plot the results
time = np.linspace(0, TOTAL_SECONDS, STEP_COUNT+1)
target_angles = [TARGET_ANGLE] * (STEP_COUNT+1)
plt.figure()
plt.plot(time, solved_angles, label='angle (rad)')
plt.plot(time, target_angles, label='target angle (rad)')
# plt.plot(time[:-1], solved_control_voltages, label='voltage')
# plt.plot(time[:-1], solved_control_currents, label='current')
plt.xlabel('Time [s]')
plt.legend(loc='upper right')
plt.show()
