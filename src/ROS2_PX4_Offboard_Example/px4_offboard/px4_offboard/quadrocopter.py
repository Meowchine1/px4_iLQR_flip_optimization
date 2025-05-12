import numpy as np

class MultirotorDynamics:
    def __init__(self):
        self.vehicle_mass = 0.820  # kg
        self.motor_time_constant = 0.02  # sec
        self.motor_rotational_inertia = 6.56e-6  # kg·m²
        self.thrust_coefficient = 1.48e-6  # N/(rad/s)^2
        self.torque_coefficient = 9.4e-8  # Nm/(rad/s)^2
        self.drag_coefficient = 0.1  # N/(m/s)

        self.aeromoment_coefficient = np.diag([0.003, 0.003, 0.003])  # Nm/(rad/s)^2

        self.vehicle_inertia = np.diag([0.045, 0.045, 0.045])  # kg·m²

        self.max_prop_speed = 2100  # rad/s
        self.min_prop_speed = 0.0  # rad/s

        self.moment_process_noise = 1.25e-7  # (Nm)^2·s
        self.force_process_noise = 0.0005  # N^2·s

        self.moment_arm = 0.15  # m

        # IMU parameters
        self.acc_bias_process_noise = 0.0
        self.gyro_bias_process_noise = 0.0
        self.acc_bias_init_var = 0.00001
        self.gyro_bias_init_var = 0.00001
        self.acc_noise_var = 0.0001
        self.gyro_noise_var = 0.00001

        self.gravity = np.array([0.0, 0.0, -9.81])  # m/s²

        # Initial state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.array([1, 0, 0, 0])  # Quaternion [w, x, y, z]
        self.angular_velocity = np.zeros(3)
        self.prop_speeds = np.ones(4) * np.sqrt(self.vehicle_mass * 9.81 / 4 / self.thrust_coefficient)

    def set_initial_position(self, position: np.ndarray, attitude: np.ndarray):
        self.position = np.array(position)
        self.attitude = np.array(attitude)

    def process(self, dt: float, motor_setpoints: list):
        self.prop_speeds = np.clip(np.array(motor_setpoints), self.min_prop_speed, self.max_prop_speed)
        # Здесь можно вызвать динамику мультикоптера: интеграция, силы, моменты и т.д.

    def get_vehicle_position(self):
        return self.position

    def get_vehicle_velocity(self):
        return self.velocity

    def get_vehicle_attitude(self):
        return self.attitude

    def get_vehicle_angular_velocity(self):
        return self.angular_velocity

    def get_motors_rpm(self):
        return self.prop_speeds * (60 / (2 * np.pi))

    def get_imu_measurement(self):
        # Генерация показаний ИНС (упрощённо, без шума и смещений)
        acc_meas = self.gravity
        gyro_meas = self.angular_velocity
        return acc_meas, gyro_meas
