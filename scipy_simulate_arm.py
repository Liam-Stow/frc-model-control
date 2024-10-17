import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import arm_dynamics

# Parameters
TIME_STEP = 0.05
TOTAL_SECONDS: float = 5.0
STEP_COUNT = int(TOTAL_SECONDS / TIME_STEP)
initial_angle = 0  # initial angle (pointed straight to the right)
initial_velocity = 0 # initial velocity (not moving)

# Solve the ODE using solve_ivp
solution = solve_ivp(
    lambda t,state: arm_dynamics.calc_derivatives(state[0], state[1], 0),
    [0.0, TOTAL_SECONDS], 
    [initial_angle, initial_velocity], 
    t_eval=np.linspace(0, TOTAL_SECONDS, STEP_COUNT+1),  # Time points for evaluation
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