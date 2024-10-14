import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
G = -9.81  # gravity
L = 1.0    # length of the pendulum
M = 2.0    # mass of the pendulum
T = 25.0   # total simulation time
initial_angle = -2 * np.pi  # initial angle (pointed down)
initial_velocity = 0        # initial velocity (not moving)

# Define the system dynamics
def pendulum_dynamics(t, y):
    angle, velocity = y
    # Equation of motion: d²θ/dt² = (g/L) * cos(θ)
    d_angle_dt = velocity
    d_velocity_dt = (G / L) * np.cos(angle)
    return [d_angle_dt, d_velocity_dt]

# Solve the ODE using solve_ivp
solution = solve_ivp(
    pendulum_dynamics, 
    [0, T], 
    [initial_angle, initial_velocity], 
    t_eval=np.linspace(0, T, 500),  # Time points for evaluation
    method='RK45'
)

# Extract the results
solved_angles = solution.y[0]
solved_velocities = solution.y[1]
time = solution.t

# Plot the results
plt.figure()
plt.plot(time, solved_angles, label='angle (rad)')
plt.plot(time, solved_velocities, label='velocity (rad/s)')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend(loc='upper right')
plt.title('Arm Motion Without Control Torques')
plt.show()