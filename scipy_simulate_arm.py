import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import arm_dynamics
import json

# Parameters
TIME_STEP = 0.005
TOTAL_SECONDS: float = 0.5
STEP_COUNT = int(TOTAL_SECONDS / TIME_STEP)
initial_angle = -np.pi/2.0  # initial angle (pointed straight down)
initial_velocity = 0 # initial velocity (not moving)

# load a control strategy
strategy = json.loads(open('sleipnir_arm_strategy.json').read())
def get_control(strategy: list[dict], time: float) -> dict:
    for s in strategy:
        if s['time'] >= time:
            return s
    print('No control found for time', time)
    return strategy[-1]

# Simulate the arm
solution = solve_ivp(
    lambda t,state: arm_dynamics.calc_derivatives(state[0], state[1], get_control(strategy, t).get('control_torque', 0.0)),
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