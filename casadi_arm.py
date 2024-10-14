import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
target_angle = 0
G = -9.81  # gravity
L = 1.0   # length of the pendulum
M = 2.0   # mass of the pendulum
dt = 0.05  # time step
N = 500   # number of control intervals
T = N * dt  # total time

# Decision variables for state trajectory and control inputs
opti = ca.Opti()
states = opti.variable(2, N+1)
angles = states[0, :]
velocities = states[1, :]
control_torques = opti.variable(1, N)

# Objective function (minimize distance from target angle)
opti.minimize(ca.sumsqr(angles - target_angle))

# Constraints
opti.subject_to(angles[0] == -2*np.pi)  # initial angle (pointed down)
opti.subject_to(velocities[0] == 0)  # initial velocity (not moving)

for k in range(N):
    angle = angles[k]
    velocity = velocities[k]
    control_torque = control_torques[k]
    next_angle = angles[k+1]
    next_velocity = velocities[k+1]

    opti.subject_to(next_angle == angle + velocity * dt)
    opti.subject_to(next_velocity == velocity + (G/L) * ca.cos(angle) * dt + control_torque / (M * L**2) * dt)

opti.subject_to(opti.bounded(-20, control_torques, 20)) 
opti.subject_to(angles[-1] == target_angle)  # final angle = target
opti.subject_to(velocities[-1] == 0)  # final velocity = 0

# Solve
opti.solver('ipopt')
solution = opti.solve()

# Extract the solution
solved_angles = solution.value(angles)
solved_velocities = solution.value(velocities)
solved_control_torques = solution.value(control_torques)

# Plot the results
time = np.linspace(0, T, N+1)

target_angles = [target_angle] * (N+1)
plt.figure()
plt.plot(time, solved_angles, label='angle (rad)')
plt.plot(time, target_angles, label='target angle (rad)')
plt.plot(time[:-1], solved_control_torques, label='control (N)')
plt.xlabel('Time [s]')
plt.ylabel('States / Control')
plt.legend(loc='upper right')
plt.show()
