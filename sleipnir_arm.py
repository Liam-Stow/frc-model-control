from jormungandr.optimization import OptimizationProblem
from jormungandr.autodiff import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import math

total_time = 5.0  # s
time_step = 0.005  # 5 ms
step_count = int(total_time / time_step)

gravity = -9.8 # m/s^2
mass = 2.0 # kg
length = 2.0 # m
initial_angle = 0.0 # rad 

problem = OptimizationProblem()

states = problem.decision_variable(step_count+1, 2) # Matrix: (angle, rotational velocity) x simulation steps
efforts = problem.decision_variable(step_count, 1) # Matrix: (torque effort) x simulation steps
angles = states[:, 0]
vels = states[:, 1]

steps = [
    {
        'angle': angles[i], 
        'vel': vels[i], 
        'next_angle': angles[i+1], 
        'next_vel': vels[i+1], 
        'effort': efforts[i],
        'target': math.pi/2 #math.cos(i/100)
     } for i in range(step_count)
]

for step in steps:
    angle = step['angle']
    vel = step['vel']
    next_angle = step['next_angle']
    next_vel = step['next_vel']
    effort = step['effort']

    grav_accel = gravity/length*cos(angle)
    effort_accel = effort/(mass*length**2)
    friction_accel = -vel/2.0
    problem.subject_to(next_angle == angle + vel * time_step)
    problem.subject_to(next_vel == vel + grav_accel * time_step + friction_accel * time_step + effort_accel * time_step)

# Constraints
problem.subject_to(angles[0] == initial_angle)
problem.subject_to(vels[0] == 0.0)

# problem.subject_to(vels <= 2.5)
# problem.subject_to(vels >= -2.5)

problem.subject_to(efforts <= 40.0)
problem.subject_to(efforts >= -40.0)

# Cost func
cost = 0.0
for step in steps:
    cost += (step['target'] - step['angle'])**2
    step['cost'] = cost

problem.minimize(cost)

problem.solve()


solved_poss = [p.value() for p in angles]
solved_vels = [v.value() for v in vels]
solved_efforts = [e.value() for e in efforts]
solved_costs = [s['cost'].value() for s in steps]

plt.plot(solved_poss[:-1])
# plt.plot(solved_vels)
# plt.plot(solved_efforts)
plt.plot([s['target'] for s in steps])
