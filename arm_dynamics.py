from numpy import cos as npcos
from jormungandr.autodiff import cos as jcos
from jormungandr.autodiff import Variable as jvar

GRAVITY = -9.81  
LENGTH = 1.0 
MASS = 2.0

def calculate_derivatives_no_control(angle, velocity):
    return calculate_derivatives(angle, velocity, 0)

# Calculate derivatives
def calculate_derivatives(angle, velocity, control_torque):
    cos = jcos if type(angle) == jvar else npcos
    FRICTION_COEFFICIENT = 3.0
    d_angle_dt = velocity
    d_velocity_dt = (
        (GRAVITY / LENGTH) * cos(angle)
        - FRICTION_COEFFICIENT * velocity
        + control_torque / (MASS * LENGTH**2)
    )
    return d_angle_dt, d_velocity_dt

# Calculate next state
def next_state(angle, velocity, control_torque, dt):
    d_angle_dt, d_velocity_dt = calculate_derivatives(angle, velocity, control_torque)
    angle = angle + d_angle_dt * dt
    velocity = velocity + d_velocity_dt * dt
    return angle, velocity
