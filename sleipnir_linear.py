from jormungandr.optimization import OptimizationProblem
import numpy as np
import matplotlib.pyplot as plt

total_time = 5.0  # s
time_step = 0.005  # 5 ms
step_count = int(total_time / time_step)
target_distance = 2 # m

problem = OptimizationProblem()

states = problem.decision_variable(step_count + 1, 2) # 2 x step_count+1 matrix to hold a state for every time step in the sim
inputs = problem.decision_variable(step_count, 1) # 1 x step_count matrix to hold the input value for every time step in the sim (last input of the sequence doens't matter)
poss = states[:, 0]
vels = states[:, 1]

sim_steps = [
    {
        'pos': poss[i], 
        'vel': vels[i], 
        'next_pos': poss[i+1], 
        'next_vel': vels[i+1], 
        'effort': inputs[i]
     } for i in range(step_count)
]

for step in sim_steps:
    position = step['pos']
    velocity = step['vel']
    acceleration = step['effort']
    
    next_position =  step['next_pos']
    next_velocity = step['next_vel'] 

    # Next position is determiend by the kinematic equations
    problem.subject_to(next_position == position + velocity * time_step + 0.5 * acceleration * time_step**2)

    # Next velocity is current vel + accel
    problem.subject_to(next_velocity == velocity + acceleration * time_step)



# Start and end conditions
problem.subject_to(vels[0] == 0.0)
problem.subject_to(poss[0] == 0.0)

# limit vel
problem.subject_to(-1 <= vels)
problem.subject_to(1 >= vels)

# limit accel
problem.subject_to(-1 <= inputs)
problem.subject_to(1 >= inputs)


# Cost function 
cost = 0.0
for step in sim_steps:
    cost += (target_distance - step['pos']) ** 2
problem.minimize(cost)

problem.solve()


solved_poss = [p.value() for p in poss]
solved_vels = [v.value() for v in vels]
solved_efforts = [e.value() for e in inputs]

plt.plot(solved_poss)
plt.plot(solved_vels)
plt.plot(solved_efforts)