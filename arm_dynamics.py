from numpy import cos as npcos
from jormungandr.autodiff import cos as jcos
from jormungandr.autodiff import Variable as jvar

GRAVITY = -9.81  
LENGTH = 1.0 
MASS = 2.0
COM_DISTANCE = LENGTH/2.0
# Motor is kraken with FOC
GEARING = 10.0
MOTOR_RESISTANCE = 0.025 # ohms
MOTOR_KV = 182765.294 # rad/s/volt
MOTOR_KT = 0.0194 # Nm / A


def calc_current(voltage, velocity):
    return (
        -1.0 / (MOTOR_KV/GEARING) / MOTOR_RESISTANCE * velocity
        + 1.0 / MOTOR_RESISTANCE * voltage
    )

def calc_torque(current):
    return MOTOR_KT * current * GEARING

def calc_derivatives_with_voltage(angle, velocity, voltage):
    current = calc_current(voltage, velocity)
    torque = calc_torque(current)
    return calc_derivatives(angle, velocity, torque)

def calc_derivatives(angle, velocity, control_torque):
    cos = jcos if type(angle) == jvar else npcos # use jormungandr's cos() if jormungandr variables are used
    FRICTION_COEFFICIENT = 0.0
    d_angle_dt = velocity
    d_velocity_dt = (
        (GRAVITY / COM_DISTANCE) * cos(angle)
        - FRICTION_COEFFICIENT * velocity
        + control_torque / (MASS * COM_DISTANCE**2)
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
