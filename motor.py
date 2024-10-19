class Motor:
    def __init__(self, nominal_voltage, stall_torque, stall_current, free_current, free_speed, num_motors=1):
        self.nominal_voltage = nominal_voltage
        self.stall_torque = stall_torque * num_motors
        self.stall_current = stall_current * num_motors
        self.free_current = free_current * num_motors
        self.free_speed = free_speed
        self.resistance = nominal_voltage / self.stall_current
        self.Kv = free_speed / (nominal_voltage - self.resistance * self.free_current)
        self.Kt = self.stall_torque / self.stall_current

    def calc_current_from_velocity_voltage(self, velocity, voltage):
        return -1.0 / self.Kv / self.resistance * velocity + 1.0 / self.resistance * voltage

    def calc_current_from_torque(self, torque):    
        return torque / self.Kt

    def calc_torque_from_current(self, current):    
        return self.Kt * current

    def calc_torque_from_velocity_voltage(self, velocity, voltage):
        return self.calc_torque_from_current(
            self.calc_current_from_velocity_voltage(velocity, voltage)
        )

    def calc_voltage_from_torque_velocity(self, torque, velocity):
        return -1.0 / self.Kv * velocity + 1.0 / self.Kt * self.resistance * torque

    def calc_velocity_from_voltage_torque(self, voltage, torque):
        return voltage * self.Kv - 1.0 / self.Kt * torque * self.resistance * self.Kv

    def with_reduction(self, gearing_reduction):
        return Motor(
            self.nominal_voltage,
            self.stall_torque * gearing_reduction,
            self.stall_current,
            self.free_current,
            self.free_speed / gearing_reduction
        )

    def KrakenX60FOC(num_motors=1):
        return Motor(12, 9.37, 483, 2, 5800, num_motors=num_motors)
