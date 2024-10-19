from numpy import cos as npcos
from jormungandr.autodiff import cos as jcos
from jormungandr.autodiff import Variable as jvar
from motor import Motor

GRAVITY = -9.81  
LENGTH = 1.0 
MASS = 2.0
COM_DISTANCE = LENGTH/2.0
MOI = 1.0/3.0 * MASS * LENGTH**2
GEARING = 10.0
motor = Motor.KrakenX60FOC().with_reduction(GEARING)

def calc_derivatives_with_voltage(angle, velocity, voltage):
    current = motor.calc_current_from_velocity_voltage(velocity, voltage)
    torque = motor.calc_torque_from_current(current)
    return calc_derivatives(angle, velocity, torque)

def calc_derivatives(angle, velocity, control_torque):
    cos = jcos if type(angle) == jvar else npcos # use jormungandr's cos() if jormungandr variables are used
    FRICTION_COEFFICIENT = 0.0
    d_angle_dt = velocity
    d_velocity_dt = (
        GRAVITY * MASS * cos(angle) / MOI
        - FRICTION_COEFFICIENT * velocity
        + control_torque / MOI
    )
    return d_angle_dt, d_velocity_dt

def next_state_with_torque(angle, velocity, control_torque, dt):
    d_angle_dt, d_velocity_dt = calc_derivatives(angle, velocity, control_torque)
    angle = angle + d_angle_dt * dt
    velocity = velocity + d_velocity_dt * dt
    return angle, velocity

def next_state_with_voltage(angle, velocity, control_voltage, dt):
    d_angle_dt, d_velocity_dt = calc_derivatives_with_voltage(angle, velocity, control_voltage)
    angle = angle + d_angle_dt * dt
    velocity = velocity + d_velocity_dt * dt
    return angle, velocity
