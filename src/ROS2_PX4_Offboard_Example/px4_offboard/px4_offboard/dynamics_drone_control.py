import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import (VehicleAttitude, VehicleImu, ActuatorOutputs, ActuatorMotors, 
                          VehicleLocalPosition,SensorCombined,VehicleAngularVelocity, 
                          VehicleAngularAccelerationSetpoint, VehicleMagnetometer, SensorBaro, EscStatus)
import numpy as np

 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
 
from filterpy.kalman import ExtendedKalmanFilter
from datetime import datetime
from sensor_msgs.msg import Imu, MagneticField, FluidPressure
import os

import csv
import time
from std_msgs.msg import String
from datetime import datetime
   
from quad_flip_msgs.msg import OptimizedTraj
from rclpy.qos import QoSProfile

import threading
from jax import jit, grad, jacobian, hessian, vmap, lax
import jax.numpy as jnp 
from scipy.spatial.transform import Rotation as Rot 
 

# ============ CONSTANTS ====================================================
SEA_LEVEL_PRESSURE = 101325.0
EKF_DT = 0.01
# ============ DRONE CONSTRUCT PARAMETERS ===================================
MASS = 0.82
INERTIA = np.diag([0.045, 0.045, 0.045])
ARM_LEN = 0.15
K_THRUST = 1.48e-6
K_TORQUE = 9.4e-8
MOTOR_TAU = 0.02
MAX_SPEED = 2100.0
DRAG = 0.1
MAX_RATE = 25.0  # ограничение на угловую скорость (roll/pitch) рад/с

# ============ Гиперпараметры для ModelPredictiveController =========================================
dt = 0.1
horizon = 10  # Горизонт предсказания
n = 13        # Размерность состояния квадрокоптера (позиция, скорость, ориентация, угловая скорость) 
m = 4         # Размерность управления (4 мотора)

# ****************** Настройка стоимостей iLQR ******************
Q = jnp.diag(jnp.array([
    1.0, 1.0, 10.0,       # x, y — менее важны, z — важна
    1.0, 1.0, 1.0,        # vx, vy, vz
    0.0, 50.0, 50.0, 0.0, # ориентация
    5.0, 5.0, 1.0         # угловые скорости
]))

R = jnp.diag(jnp.array([
        0.001, 0.001, 0.001, 0.001  # все моторы слабо штрафуются
    ]))

Qf = jnp.diag(jnp.array([
    1.0, 1.0, 10.0,       # позиции: x, y — меньше важны, z — важна
    0.1, 0.1, 0.1,        # скорости
    0.0, 100.0, 100.0, 0.0, # ориентация (qx, qy)
    10.0, 10.0, 1.0       # угловые скорости
]))
 
# ===== MATRIX OPERTIONS =====
# QUATERNION UTILS (SCIPY-based)
def quat_to_rot_matrix_numpy(quat):
    # Кватернион: [w, x, y, z]
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return R

def quat_multiply_numpy(q, r):
    # Кватернионы [w, x, y, z]
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])
 
def f_numpy(x, u, dt):
    m = MASS
    I = INERTIA
    arm = ARM_LEN
    kf = K_THRUST
    km = K_TORQUE
    drag = DRAG
    g = np.array([0.0, 0.0, 9.81])
    max_speed = MAX_SPEED

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]

    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quat = quat / quat_norm

    R_bw = quat_to_rot_matrix_numpy(quat)

    rpm = np.clip(u, 0.0, max_speed)
    w_squared = rpm ** 2
    thrusts = kf * w_squared

    Fz_body = np.array([0.0, 0.0, np.sum(thrusts)])
    F_world = R_bw @ Fz_body - m * g - drag * vel
    acc = F_world / m

    new_vel = vel + acc * dt
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2

    tau = np.array([
        arm * (thrusts[1] - thrusts[3]),
        arm * (thrusts[2] - thrusts[0]),
        km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3])
    ])

    omega_cross = np.cross(omega, I @ omega)
    omega_dot = np.linalg.solve(I, tau - omega_cross)
    new_omega = omega + omega_dot * dt

    omega_quat = np.concatenate(([0.0], new_omega))
    dq = 0.5 * quat_multiply_numpy(quat, omega_quat)
    new_quat = quat + dq * dt
    new_quat /= np.linalg.norm(new_quat) + 1e-8  # безопасная нормализация

    x_next = np.concatenate([new_pos, new_vel, new_quat, new_omega])
    return x_next

@jit
def quat_multiply(q1, q2):
    """
    Умножение кватернионов q1 * q2
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])

@jit
def quat_to_rot_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return jnp.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

@jit
def f(x, u, dt):
    m = MASS
    I = INERTIA
    arm = ARM_LEN
    kf = K_THRUST
    km = K_TORQUE
    drag = DRAG
    g = jnp.array([0.0, 0.0, 9.81])
    max_speed = MAX_SPEED

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]

    # нормализация кватерниона через jax.lax.cond
    quat_norm = jnp.linalg.norm(quat)
    quat = lax.cond(
        quat_norm < 1e-8,
        lambda _: jnp.array([1.0, 0.0, 0.0, 0.0]),
        lambda _: quat / quat_norm,
        operand=None
    )

    R_bw = quat_to_rot_matrix(quat)
    rpm = jnp.clip(u, 0.0, max_speed)
    w_squared = rpm ** 2
    thrusts = kf * w_squared

    Fz_body = jnp.array([0.0, 0.0, jnp.sum(thrusts)])
    F_world = R_bw @ Fz_body - m * g - drag * vel

    acc = F_world / m
    new_vel = vel + acc * dt
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2

    tau = jnp.array([
        arm * (thrusts[1] - thrusts[3]), # Roll: правый - левый
        arm * (thrusts[2] - thrusts[0]), # Pitch: задний - передний
        km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3]) # Yaw
    ])

    omega_cross = jnp.cross(omega, I @ omega)
    omega_dot = jnp.linalg.solve(I, tau - omega_cross)
    new_omega = omega + omega_dot * dt

    omega_quat = jnp.concatenate([jnp.array([0.0]), new_omega])
    dq = 0.5 * quat_multiply(quat, omega_quat)
    new_quat = quat + dq * dt
    new_quat /= jnp.linalg.norm(new_quat + 1e-8)  # безопасная нормализация

    x_next = jnp.concatenate([new_pos, new_vel, new_quat, new_omega])
    return x_next

# миксер rpm --> [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
def rpm_to_control(rpm, arm, kf, km):
    w_squared = rpm ** 2
    thrusts = kf * w_squared

    # Общая тяга
    thrust = jnp.sum(thrusts)

    # Моменты по осям
    roll_torque  = arm * (thrusts[1] - thrusts[3])   # M2 - M4
    pitch_torque = arm * (thrusts[2] - thrusts[0])   # M3 - M1
    yaw_torque   = km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3])

    # Вернуть управляющие воздействия
    return jnp.array([roll_torque, pitch_torque, yaw_torque, thrust])

def rpm_to_control_normalized(rpm, arm, kf, km, max_thrust, max_torque):
    control = rpm_to_control(rpm, arm, kf, km)
    roll_cmd   = control[0] / max_torque
    pitch_cmd  = control[1] / max_torque
    yaw_cmd    = control[2] / max_torque
    thrust_cmd = control[3] / max_thrust
    return jnp.array([roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd])

class MyEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z): 
        super().__init__(dim_x, dim_z)
        self.dt = EKF_DT
        self.f = f

    def predict_x(self, u=np.zeros(4)):# predict new state with dynamic physic model
        return f_numpy(x=self.x, u=u, dt=self.dt)# custom fx(x, u, dt) function
     
class ModelPredictiveControlNode(Node):
    def __init__(self):
        super().__init__('dynamic_model_node')   
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = os.path.join("MY_LOG", self.datetime)
        self.ilqr_log_base = os.path.join("MY_iLQR_LOG", self.datetime)
        # == == == == == == == == == == == == == = PUBLISHERS = == == == == == == == == == == == == == == == == == == == == == ==
        self.server_pub = self.create_publisher(String, '/drone/server_msg', qos_profile) 
        # == == == == == == == == == == == == == =SUBSCRIBERS= == = == == == == == == == == == == == == == == == == == == == == == 
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity', self.angular_velocity_callback, qos_profile)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
        self.create_subscription(VehicleAngularAccelerationSetpoint,
        '/fmu/out/vehicle_angular_acceleration_setpoint', self.vehicle_angular_acceleration_setpoint_callback, qos_profile)
        self.create_subscription(VehicleImu,'/fmu/out/vehicle_imu',self.vehicle_imu_callback, qos_profile)

        #self.create_subscription(ActuatorOutputs, '/fmu/out/actuator_outputs', self.actuator_outputs_callback, qos_profile)
        #self.create_subscription(ActuatorMotors, '/fmu/out/actuator_motors', self.actuator_motors_callback, qos_profile) 
        
        # ****** RPM *******
        self.create_subscription(EscStatus, '/fmu/out/esc_status', self.esc_status_callback, qos_profile)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.create_subscription(SensorBaro, '/fmu/out/sensor_baro', self.sensor_baro_callback, qos_profile)
        self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.vehicle_magnetometer_callback, qos_profile)
        
        # == == == == == == == == == == == == == =DATA USED IN METHODS= == == == == == == == == == == == == == == == == == == ==  
        self.angularVelocity = np.zeros(3, dtype=np.float32)
        self.angular_acceleration = np.zeros(3, dtype=np.float32)
        self.vehicleImu_velocity_w = np.zeros(3, dtype=np.float32) # в мировых координатах 
        self.sensorCombined_linear_acceleration = np.zeros(3, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32) # drone position estimates with IMU localization
        #self.motor_inputs = np.zeros(4, dtype=np.float32)  #приближение в радианах
        #self.motor_rpms = np.zeros(4) # rpm from px4 topic
        self.motor_rpms = jnp.zeros(4)
        self.vehicleAttitude_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # quaternion from topic
        self.magnetometer_data = np.zeros(3, dtype=np.float32)
        self.baro_pressure = 0.0
        self.baro_altitude = 0.0
        self.mag_yaw = 0.0
        #self.actuator_motors = np.zeros(4)
        
        # FOR SITL TESTING  
        self.vehicleLocalPosition_position = np.zeros(3, dtype=np.float32)
  
        # =================================== OTHER TOPIC DATA ========================================================= 
        # [TOPIC NAME]_[PARAM NAME] OR [TOPIC NAME] IF PARAM = TOPIC NAME
        self.sensorCombined_angular_velocity = np.zeros(3, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.zeros(3, dtype=np.float32)
        self.baro_temperature = 0.0 # temperature in degrees Celsius
        # =================================== Гиперпараметры для EKF ===================================================
        #self.new_x=np.zeros(13)
        # * вектор состояния 13 штук: позиция, скорость, ориентация (4), угловые скорости 
        # * вектор измерений 14 штук: позиция, линейная скорость, ориентация (4), барометрическая высота  
        #self.ekf = MyEKF(dim_x=13, dim_z=13)
        #self.ekf.x = np.zeros(13)
        self.measurnments = np.zeros(13)
        self.z= np.zeros(13)
        #self.ekf.x[6] = 1.0  # qw = 1 (единичный кватернион)
        # * Covariance matrix
        # Ковариация состояния
        #self.ekf.P *= 0.1

        # Процессный шум
        # self.ekf.Q = np.diag([
        #     0.001, 0.001, 0.001,         # x, y, z
        #     0.01, 0.01, 0.01,            # vx, vy, vz
        #     0.0001, 0.0001, 0.0001, 0.0001,  # qw, qx, qy, qz
        #     0.00001, 0.00001, 0.00001        # wx, wy, wz
        # ])
        # Измерительный шум (z не используется из позиции, вместо него — баро)
        # self.ekf.R = np.diag([
        #     0.1, 0.1,                    # позиция x, y (м²)
        #     0.0001, 0.0001, 0.0001,      # скорость vx, vy, vz
        #     0.00001, 0.00001, 0.00001, 0.00001,  # qw, qx, qy, qz
        #     0.00001, 0.00001, 0.00001,   # wx, wy, wz
        #     0.5                          # баро (вместо позиции z)
        # ])
        #    ====    ====   Параметры ModelPredictiveController    ====     ====     ====     ====     ====     ====     ====
        self.optimizer = ILQROptimizer(
            logger=self.get_logger()) 

        self.phase = 'init'
        self.takeoff_altitude = -5.0  # м TODO DOES IT WOULD BE NEGATIVE?
        self.takeoff_tol = 0.1
        self.flip_started_time = None
        self.flip_duration = 1.0  # с, продолжительность флипа
        self.recovery_time = 2.0  # с, стабилизация после флипа
        self.recovery_start_time = None
        self.landing_altitude = 0.2  # м
        self.roll_abs_tol = 0.1  # допуск 0.1 рад
        #   ====    ====    ====     ====     ====     ====     ====     ====     ====     ====     ====

        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_file_name_pos = f"{now_str}_pos.log"
        self.log_file_name_quat = f"{now_str}_quat.log"
        self.log_file_name_vel = f"{now_str}_vel.log"
        self.log_file_name_ang_vel = f"{now_str}_ang_vel.log"

         # ======= TIMERS =======
        #self.timer = self.create_timer(0.01, self.step_dynamics)
        self.EKF_timer = self.create_timer(EKF_DT, self.EKF)  
        self.mpc_controller = self.create_timer(0.05, self.mpc_control_loop)
         
        # == == == =CLIENT SERVER INTERACTION= == == == 
        self.create_subscription(String, '/drone/client_msg', self.client_msg_callback, qos_profile)#
        self.pub_optimized_traj = self.create_publisher(OptimizedTraj, '/drone/optimized_traj', qos_profile)

        self.optimized_traj_f = False
        self.X_opt = np.zeros((horizon + 1, n))  # (N+1) x n
        self.u_optimal = np.zeros((horizon, m))  # N x m
        self.i_final = 0
        self.cost_final = 0.0
        self.done = False 
        self.to_client_f = False
        self.mpc_lock = threading.Lock()
        #self.x0 = self.ekf.x.copy()
        #self.u_init = jnp.tile(self.actuator_motors, (horizon, 1))
        self.x_target_traj = jnp.zeros((horizon + 1, 13))
        self.u_target_traj = jnp.tile(self.motor_rpms, (horizon, 1))
        self.current_time = self.get_clock().now().nanoseconds * 1e-9
                
    # def actuator_outputs_callback(self, msg: ActuatorOutputs):# TODO SHOULDNOT USE IT
    #     pwm_outputs = msg.output[:4]  # предполагаем, что 0-3 — это моторы
    #     # преобразование PWM в радианы в секунду (линейное приближение)
    #     self.motor_inputs = np.clip((np.array(pwm_outputs) - 1000.0) / 1000.0 * MAX_SPEED, 0.0, MAX_SPEED)
       
    # def actuator_motors_callback(self, msg: ActuatorMotors):# TODO SHOULDNOT USE IT
    #     self.actuator_motors = np.sqrt(np.clip(msg.control[:4], 0.0, None) / K_THRUST)
       
    def esc_status_callback(self, msg: EscStatus):
        rpms = [esc.esc_rpm for esc in msg.esc[:msg.esc_count]]
        self.motor_rpms = np.clip(np.array(rpms), 0.0, MAX_SPEED)
        #self.get_logger().info(f"dyn esc_status_callback self.motor_rpms: {self.motor_rpms}")

    def send_msg_to_client(self, msg):
        server_msg = String()
        server_msg.data = msg 
        self.server_pub.publish(server_msg)

    def client_msg_callback(self, msg):
        """GET CLIENT MESSAGES"""
        command = msg.data.strip().lower()
        #self.get_logger().info(f"Received command: {command}")
        if command == "takeoff":
            self.phase = "takeoff"
            self.to_client_f = True
            self.optimized_traj_f = True
            self.send_msg_to_client("mpc_on")
        else:
            self.get_logger().warn(f"Unknown command: {command}")

    def send_optimized_traj(self):
        if self.optimized_traj_f:
            msg = OptimizedTraj()
            msg.x_opt = np.asarray(self.X_opt).flatten().astype(np.float32).tolist()
            msg.u_opt = np.asarray(self.u_optimal).flatten().astype(np.float32).tolist()
            msg.i_final = int(self.i_final)
            msg.cost_final = float(self.cost_final)
            msg.done = self.done
            self.pub_optimized_traj.publish(msg)
            
            # Логирование в CSV
            self.log_optimized_traj()
 
    def log_optimized_traj(self):
        log_base = self.log_base
        file_path = os.path.join(log_base, 'optimized_traj_log.csv')
        text_log_path = os.path.join(log_base, 'optimized_traj_log.txt')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        X_flat = np.asarray(self.X_opt[0]).flatten()
        u_flat = np.asarray(self.u_optimal).flatten()
        i_final = [self.i_final]
        cost_final = [self.cost_final]

        data = [X_flat, u_flat, i_final, cost_final]
        labels = ['X_opt', 'u_opt', 'i_final', 'cost_final']

        # --- Логирование CSV ---
        new_file = not os.path.exists(file_path)
        if new_file:
            headers = []
            for label, arr in zip(labels, data):
                if len(arr) > 1:
                    headers.extend([f"{label}[{i}]" for i in range(len(arr))])
                else:
                    headers.append(label)

            with open(file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        row_values = [float(v) for arr in data for v in arr]
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_values)

        # --- Логирование в текстовый файл ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(text_log_path, 'a') as f:
            f.write(f"=== Log time: {timestamp} ===\n")
            f.write(f"Final iteration: {self.i_final}\n")
            f.write(f"Final cost: {self.cost_final}\n")
            f.write(f"X_opt (first 10 values): {X_flat[:10]}\n")
            f.write(f"u_opt (first 10 values): {u_flat[:10]}\n\n")
 
    def quaternion_from_roll(self, roll_rad):
        r = R.from_euler('x', roll_rad)
        return r.as_quat()
        
    def roll_from_quaternion(self, q):
        """ Вычисление угла roll из кватерниона """
        qw, qx, qy, qz = q
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        return jnp.arctan2(sinr_cosp, cosr_cosp).item()

    def log_ilqr(self, data: str):
        os.makedirs(self.ilqr_log_base, exist_ok=True)
        log_path = os.path.join(self.ilqr_log_base, "ILQR_run_mpc().txt")
        with open(log_path, "a") as f:
            f.write(data + "\n")

    def log_mpc(self, data: str):
        os.makedirs(self.ilqr_log_base, exist_ok=True)
        log_path = os.path.join(self.ilqr_log_base, "MPC_target.txt")
        with open(log_path, "a") as f:
            f.write(data + "\n")
      
    def takeoff_targets(self):
        for i in range(horizon):
            pos = jnp.array(self.measurnments[0:3]).copy().at[2].set(self.takeoff_altitude)
            vel = jnp.array(self.measurnments[3:6]) 
            q = jnp.array(self.measurnments[6:10]) 
            omega = jnp.array(self.measurnments[10:13]) 
            self.x_target_traj = self.x_target_traj.at[i].set(jnp.concatenate([pos, vel, q, omega]))
            self.log_ilqr(f"====== self.motor_rpms.shape {self.motor_rpms.shape} ======")
            self.u_target_traj = self.u_target_traj.at[i].set(self.motor_rpms.copy())
        self.x_target_traj = self.x_target_traj.at[horizon].set(self.x_target_traj[horizon - 1])
        
    def flip_targets(self ):
        t_local = jnp.clip(self.current_time - self.flip_started_time, 0.0, self.flip_duration)
        roll_expected = 2 * jnp.pi * t_local / self.flip_duration
        
        # Получаем кватернион текущей ориентации
        q_current = self.measurnments[6:10]
        
        # Оценка roll из кватерниона
        self.roll_current = self.roll_from_quaternion(q_current)
        
        roll_error = roll_expected - self.roll_current
        gain_base = 0.8
        gain_adaptive = gain_base + 0.3 * jnp.tanh(roll_error)
        roll_target = self.roll_current + gain_adaptive * roll_error

        for i in range(horizon):
            alpha_i = i / horizon
            angle_i = roll_target * alpha_i
            
            pos = self.x0[0:3]
            vel = jnp.zeros(3)
            
            # Генерируем кватернион для целевой ориентации
            q = self.quaternion_from_roll(angle_i)
            
            omega_magnitude = 2 * jnp.pi / self.flip_duration + 0.2 * roll_error
            omega = jnp.array([omega_magnitude, 0.0, 0.0])
            
            self.x_target_traj = self.x_target_traj.at[i].set(jnp.concatenate([pos, vel, q, omega]))
            self.u_target_traj = self.u_target_traj.at[i].set(self.recovery_thrust.copy())

        # Оставляем последний элемент траектории неизменным
        self.x_target_traj = self.x_target_traj.at[horizon].set(self.x_target_traj[horizon - 1])
        
    def recovery_targets(self):
        t_local = jnp.clip(self.current_time - self.recovery_start_time, 0.0, self.recovery_time)
        roll_desired = 2 * jnp.pi * (1 - t_local / self.recovery_time)
        
        # Получаем кватернион текущей ориентации
        q_current = self.measurnments[6:10]
        
        # Оценка roll из кватерниона
        self.roll_current = self.roll_from_quaternion(q_current)
        
        roll_error = roll_desired - self.roll_current
        gain = 0.6 + 0.4 * (abs(roll_error) / jnp.pi)
        roll_target = self.roll_current + gain * roll_error

        for i in range(horizon):
            alpha_i = i / horizon
            angle_i = self.roll_current + alpha_i * (roll_target - self.roll_current)
            
            pos = self.x0[0:3]
            vel = jnp.zeros(3)
            
            # Генерируем кватернион для целевой ориентации
            q = self.quaternion_from_roll(angle_i)
            
            omega_mag = -2 * jnp.pi / self.recovery_time * (1 + 0.2 * abs(roll_error) / jnp.pi)
            omega = jnp.array([omega_mag, 0.0, 0.0])
            
            self.x_target_traj = self.x_target_traj.at[i].set(jnp.concatenate([pos, vel, q, omega]))
            self.u_target_traj = self.u_target_traj.at[i].set(self.recovery_thrust.copy())

        # Оставляем последний элемент траектории неизменным
        self.x_target_traj = self.x_target_traj.at[horizon].set(self.x_target_traj[horizon - 1])
        
    def land_targets(self ):
        return
      
    def run_mpc_thread(self):
        with self.mpc_lock:
            start_time = time.time()
            self.log_ilqr(f"============= phase: {self.phase}=============")
            try: 
                self.current_time = self.get_clock().now().nanoseconds * 1e-9
                if self.phase == 'takeoff':
                    self.send_msg_to_client("mpc_on")# на всякий случай если сообщене не дойдет с одного раза,
                                                     # чтобы переключить контроллер полета на прием управления траектории
                    self.takeoff_targets()
                    
                    self.log_ilqr(f"takeoff")
                    if abs( - self.takeoff_altitude) < self.takeoff_tol:
                        self.phase = 'flip'
                        self.flip_started_time = self.current_time 

                elif self.phase == 'flip': 
                    self.flip_targets()
                    self.log_ilqr(f"flip\nabs(roll_current)={abs(self.roll_current)}") 

                    if jnp.isclose(self.roll_current, 2 * jnp.pi, atol=0.1):# выражение устойчивее к шуму чем аналогичное с abs  
                        self.phase = 'recovery'
                        self.recovery_start_time = self.current_time

                elif self.phase == 'recovery':
                    self.recovery_targets()
                    if abs(self.roll_current) <= self.roll_abs_tol:
                        self.phase = 'land'

                elif self.phase == 'land':
                        self.to_client_f = False
                        self.optimized_traj_f = False
                        self.done = True
                        self.send_msg_to_client("land")

                """
                Вычисляет оптимальную траекторию состояний и управляющих воздействий
                от текущего состояния self.x0, используя iLQR.
                """ 
                self.log_mpc(f"x0:{self.measurnments}")
                self.log_mpc(f"self.motor_rpms:{self.motor_rpms}")
                self.log_mpc(f"self.x_target_traj:{self.x_target_traj}")
                self.log_mpc(f"self.u_target_traj:{self.u_target_traj}")
                # Используем ILQR для расчета оптимальной траектории


                measurnments_init = self.measurnments  # (13,)
                motor_rpms_init = jnp.tile(self.motor_rpms, (horizon, 1))  # (horizon, 4)
                X_opt, U_opt, i_final, cost_final = self.optimizer.solve( 
                    x0=measurnments_init,
                    u_init=motor_rpms_init,
                    Q=Q,
                    R=R,
                    Qf=Qf,
                    x_target_traj=self.x_target_traj,
                    u_target_traj=self.u_target_traj
                )

                self.X_opt = np.array(X_opt)          # Преобразование из jnp в np
                self.u_optimal = np.array(U_opt[0])      
                self.i_final = i_final
                self.cost_final = float(cost_final)   # Обеспечиваем float, а не jnp.scalar

                self.send_optimized_traj()
                    
            except Exception as e:
                self.log_ilqr(f"Ошибка при выполнении MPC: {str(e)}")
                # Выводим traceback ошибки для детальной диагностики
                import traceback
                self.log_ilqr(f"{traceback.format_exc()}")
            finally:
                end_time = time.time()
                elapsed = end_time - start_time
                self.log_ilqr(f"[mpc_control_loop] END phase: {self.phase}, duration: {elapsed:.3f} s")
                self.mpc_running = False
               
    def mpc_control_loop(self):
        # Запуск в отдельном потоке
        if self.optimized_traj_f:
            threading.Thread(target=self.run_mpc_thread).start()

    def ekf_logger(self):
        pos_my_ekf = self.measurnments[0:3]
        pos_real = self.vehicleLocalPosition_position

        quat_my_ekf = self.measurnments[6:10]
        px4_quat = self.vehicleAttitude_q

        vel_my_ekf = self.measurnments[3:6]
        integral_vel = self.vehicleImu_velocity_w

        omega_my_ekf = self.measurnments[10:13]
        omega_from_sensor = self.angularVelocity

        log_base = self.log_base

        # CSV файлы остаются прежними
        self._write_to_csv(
            os.path.join(log_base, 'pos_log.csv'),
            ['pos_my_ekf', 'pos_real'],
            [pos_my_ekf, pos_real],
            error_pairs=[(0,1)]
        )
        self._write_to_csv(
            os.path.join(log_base, 'quat_log.csv'),
            ['quat_my_ekf', 'px4_quat'],
            [quat_my_ekf, px4_quat],
            error_pairs=[(0, 1)]
        )
        self._write_to_csv(
            os.path.join(log_base, 'vel_log.csv'),
            ['vel_my_ekf', 'integral_vel'],
            [vel_my_ekf, integral_vel],
            error_pairs=[(0, 1)]
        )
        self._write_to_csv(
            os.path.join(log_base, 'ang_vel_log.csv'),
            ['omega_my_ekf', 'omega_from_sensor'],
            [omega_my_ekf, omega_from_sensor],
            error_pairs=[(0, 1)]
        )

        # TXT-файл только с EKF-данными
        log_txt_path = os.path.join(log_base, 'log_my_ekf.txt')
        with open(log_txt_path, 'a') as f:
            f.write('--- EKF Data ---\n')
            f.write(f'pos_my_ekf: {pos_my_ekf}\n')
            f.write(f'pos_real: {pos_real}\n')
            f.write(f'pos error: {abs(pos_real-pos_my_ekf)}\n')
            f.write(f'quat_my_ekf: {quat_my_ekf}\n') 
            f.write(f'px4_quat: {px4_quat}\n')
            f.write(f'vel_my_ekf: {vel_my_ekf}\n')
            f.write(f'omega_my_ekf: {omega_my_ekf}\n') 
            f.write(f'omega_from_sensor: {omega_from_sensor}\n')
            f.write('\n')

    def _write_to_txt(self, file_path, labels, data, error_pairs=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if error_pairs is None:
            error_pairs = []

        with open(file_path, mode='a') as f:
            # Записываем данные
            for label, values in zip(labels, data):
                formatted_values = ', '.join(f'{float(v):.6f}' for v in values)
                f.write(f"{label}: [{formatted_values}]\n")

            # Записываем ошибки
            for i, j in error_pairs:
                diff = np.array(data[i]) - np.array(data[j])
                formatted_diff = ', '.join(f'{float(v):.6f}' for v in diff)
                f.write(f"{labels[i]} - {labels[j]}: [{formatted_diff}]\n")

            f.write("\n")  # разделитель между записями

    def _write_to_csv(self, file_path, labels, data, error_pairs=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if error_pairs is None:
            error_pairs = []

        new_file = not os.path.exists(file_path)

        if new_file:
            headers = []
            for i, (label, arr) in enumerate(zip(labels, data)):
                if len(arr) > 1:
                    headers.extend([f"{label}[{j}]" for j in range(len(arr))])
                else:
                    headers.append(label)

            for i, j in error_pairs:
                label_i, label_j = labels[i], labels[j]
                arr_len = len(data[i])
                headers.extend([f"{label_i}-{label_j}[{k}]" for k in range(arr_len)])

            with open(file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        row_values = [float(v) for arr in data for v in arr]

        for i, j in error_pairs:
            diff = np.array(data[i]) - np.array(data[j])
            row_values.extend([float(v) for v in diff])

        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_values)
 
    def sensor_baro_callback(self, msg):
        #self.get_logger().info("sensor_baro_callback")
        self.baro_temperature = msg.temperature
        self.baro_pressure = msg.pressure
        self.baro_altitude = 44330.0 * (1.0 - (msg.pressure / SEA_LEVEL_PRESSURE) ** 0.1903)
        self.measurnments[2] = -self.baro_altitude
        #self.get_logger().info(f"self.baro_altitude = {self.baro_altitude}") # Все верно

    def get_yaw_from_mag(self):
        r = Rot.from_quat(self.vehicleAttitude_q)
        mag_world = r.apply(self.magnetometer_data)
        mag_x = mag_world[0]
        mag_y = mag_world[1]
        yaw_from_magnetometer = np.arctan2(-mag_y, mag_x)
        return yaw_from_magnetometer

    def vehicle_magnetometer_callback(self, msg: VehicleMagnetometer):
        #self.get_logger().info("vehicle_magnetometer_callback")
        self.magnetometer_data = np.array(msg.magnetometer_ga, dtype=np.float32)
        self.mag_yaw = self.get_yaw_from_mag()

    # ИСТИННАЯ ПОЗИЦИЯ ДЛЯ ОЦЕНКИ ИНЕРЦИАЛНОЙ ЛОКАЛИЗАЦИИ
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        #self.get_logger().info(f"vehicle_local_position_callback {msg.x} {msg.y} {msg.z}")
        self.vehicleLocalPosition_position[0] = msg.x
        self.vehicleLocalPosition_position[1] = msg.y
        self.vehicleLocalPosition_position[2] = msg.z 

    # ЛИНЕЙНОЕ УСКОРЕНИЕ, УГЛОВОЕ УСКОРЕНИЕ, КВАТЕРНИОН
    def sensor_combined_callback(self, msg: SensorCombined):
        dt_gyro = msg.gyro_integral_dt * 1e-6  # микросекунды -> секунды
        gyro_rad = np.array(msg.gyro_rad, dtype=np.float32)  # угловая скорость (рад/с)
        self.sensorCombined_angular_velocity = gyro_rad
         
        delta_angle = gyro_rad * dt_gyro # Угловое приращение (рад)
        self.sensorCombined_delta_angle = delta_angle
        self.sensorCombined_linear_acceleration = np.array(msg.accelerometer_m_s2, dtype=np.float32)
         
    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.angularVelocity = np.array(msg.xyz, dtype=np.float32)
        self.angularVelocity_angular_acceleration = np.array(msg.xyz_derivative, dtype=np.float32)
        #self.new_x[10:13] = self.angularVelocity
        # хорошая
        #self.get_logger().info(f"self.angularVelocity {self.angularVelocity[0]} {self.angularVelocity[1]} {self.angularVelocity[2]}")

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        # In this system we use scipy format for quaternion. 
        # PX4 topic uses the Hamilton convention, and the order is q(w, x, y, z). So we reorder it
        self.vehicleAttitude_q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=np.float32)
        
    def vehicle_angular_acceleration_setpoint_callback(self, msg: VehicleAngularAccelerationSetpoint):
        self.angular_acceleration = msg.xyz
   
    def vehicle_imu_callback(self, msg: VehicleImu):
        delta_velocity = np.array(msg.delta_velocity, dtype=np.float32)  # м/с
        delta_velocity_dt = msg.delta_velocity_dt * 1e-6  # с
        # Проверяем наличие ориентации и валидного времени интеграции
        if delta_velocity_dt > 0.0:
            rotation = Rot.from_quat(self.vehicleAttitude_q)
            delta_velocity_world = rotation.apply(delta_velocity)
            gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
            delta_velocity_world += gravity * delta_velocity_dt
            self.vehicleImu_velocity_w += delta_velocity_world
            self.position += self.vehicleImu_velocity_w * delta_velocity_dt
            
 
    def publish_motor_inputs(self):
        msg = Float32MultiArray()
        msg.data = self.motor_inputs.tolist()
        self.motor_pub.publish(msg)

    def log_ekf_measurements_txt(self):
        log_file = os.path.join(self.log_base, 'ekf_measurements_log.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        labels = [
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'qw', 'qx', 'qy', 'qz',
            'wx', 'wy', 'wz'
        ]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(log_file, mode='a') as f:
            f.write(f"[{timestamp}] ")
            for label, val in zip(labels, self.z):
                f.write(f"{label}={val:.6f} ")
            f.write("\n")

    def EKF(self):
        """ Основная функция обновления фильтра Калмана. """
        self.z = np.array([
            self.position[0],    # x
            self.position[1],    # y
            -self.baro_altitude, # высота по барометру TODO  BARO IS NEG
            self.vehicleImu_velocity_w[0],      # vx
            self.vehicleImu_velocity_w[1],      # vy
            self.vehicleImu_velocity_w[2],      # vz
            self.vehicleAttitude_q[0],  # qw   
            self.vehicleAttitude_q[1],  # qx
            self.vehicleAttitude_q[2],  # qy
            self.vehicleAttitude_q[3],  # qz
            self.angularVelocity[0],   # wx
            self.angularVelocity[1],   # wy
            self.angularVelocity[2]    # wz 
        ])
        self.measurnments = self.z.copy()
        #self.ekf.x = self.ekf.predict_x(self.motor_rpms)
        #self.ekf.update(z, HJacobian=self.HJacobian, Hx=self.hx) 
        # LOG RESULTS
        self.ekf_logger()
        self.log_ekf_measurements_txt()

        # Проверка выхода динамической модели
        x_next = f_numpy(x=self.z, u=np.array(self.motor_rpms, dtype=np.float32), dt=dt)
  
    def hx(self, x):
        """ Модель измерений: что бы показали датчики при текущем состоянии. """
        return np.array([
            x[0],  # x
            x[1],  # y
            x[2], # baro z
            x[3],  # vx
            x[4],  # vy
            x[5],  # vz
            x[6],  # qw
            x[7],  # qx
            x[8],  # qy
            x[9],  # qz
            x[10], # wx
            x[11], # wy
            x[12] # wz 
        ])

    def HJacobian(self, x):
        """ Якобиан модели измерений. """
        H = np.zeros((13, 13))  # 13 измерений на 13 состояний
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 3] = 1.0  # vx
        H[3, 4] = 1.0  # vy
        H[4, 5] = 1.0  # vz
        H[5, 6] = 1.0  # qw
        H[6, 7] = 1.0  # qx
        H[7, 8] = 1.0  # qy
        H[8, 9] = 1.0  # qz
        H[9, 10] = 1.0  # wx
        H[10, 11] = 1.0  # wy
        H[11, 12] = 1.0  # wz
        H[12, 2] = 1.0   # z (барометр)
        return H
 
class ILQROptimizer:
    def __init__(self, logger): 
        self.logger = logger
        self.datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = os.path.join("MY_iLQR_LOG", self.datetime)
        self.last_log_time = time.time()  # инициализация
        self.last_reduced_log_time =  time.time()

    def log_ilqr_reduced(self, data:str):
        os.makedirs(self.log_base, exist_ok=True)
        now = time.time()
        elapsed_ms = (now - self.last_reduced_log_time) * 1000  # в миллисекундах
        self.last_log_time = now
        log_path = os.path.join(self.log_base, "ILQR_solve()_reduced.txt")
        
        with open(log_path, "a") as f:
            f.write(f"[+{elapsed_ms:.1f}ms] {data}\n")

    def log_ilqr(self, data: str):
        os.makedirs(self.log_base, exist_ok=True)
        now = time.time()
        elapsed_ms = (now - self.last_log_time) * 1000  # в миллисекундах
        self.last_log_time = now
        log_path = os.path.join(self.log_base, "ILQR_solve().txt")
        
        with open(log_path, "a") as f:
            f.write(f"[+{elapsed_ms:.1f}ms] {data}\n")

    def solve(self,  x0,  u_init,  x_target_traj,  u_target_traj, Q, R, Qf,
              max_iters=5, tol=1e-3, alpha=1.0):
        X =  simulate_trajectory(x0, u_init)
        #self.log_ilqr(f"X=self.simulate_trajectory(self.x0, self.u_init): X={X}") 
        U =  u_init
        self.log_ilqr_reduced(f"start solve x_target_traj.shape {x_target_traj.shape}")
        for i in range(max_iters):
            self.log_ilqr(f"======= i={i}=======\ncost_function_traj_flip start")

            cost_prev = cost_function_traj_flip(X, U,  x_target_traj,  u_target_traj, Q, R, Qf)
            self.log_ilqr("cost_function_traj_flip end\nbackward_pass start") 

            K_list, k_list = backward_pass(X, U,  x_target_traj,  u_target_traj, Q, R, Qf)
            self.log_ilqr("backward_pass end\nforward_pass start") 

            X_new, U_new = forward_pass(X, U, k_list, K_list, alpha)
            self.log_ilqr("forward_pass end\ncost_function_traj_flip start") 

            cost_new = cost_function_traj_flip(X_new, U_new, x_target_traj, u_target_traj, Q, R, Qf)
            self.log_ilqr("cost_function_traj_flip end") 

            if jnp.abs(cost_prev - cost_new) < tol:
                break
            X, U = X_new, U_new
        self.log_ilqr(f"************end solve*************")
        self.log_ilqr_reduced(f"end solve X.shape={X.shape}")
        return X, U, i, cost_new
 

"""
f: функция динамики: f(x, u, dt) → x_next
fx_batch: функция для вычисления Якобианов A, B по заданному состоянию и управлению
dt: шаг по времени
horizon: длина горизонта предсказания
Q, R, Qf: матрицы весов стоимости
n: размерность состояния
m: размерность управления
""" 
@jit
def simulate_trajectory(x0, U):# распараллеленная
    X = [x0]
    # Используем vmap для параллельного вычисления следующего состояния
    f_batch = vmap(f, in_axes=(None, 0, None))  # Здесь U — это батч управляющих сигналов
    U = jnp.array(U)
    X_new = f_batch(x0, U, dt)
    X.extend(X_new)
    return jnp.stack(X)

@jit
def linearize_dynamics(x, u):
    A = jacobian(f, argnums=0)(x, u, dt)
    B = jacobian(f, argnums=1)(x, u, dt)
    return A, B

@jit
def quadratize_cost(x, u, x_target, u_target, Q, R):
    def scalar_cost(x_, u_):
        dx = x_ - x_target
        du = u_ - u_target
        return dx @ Q @ dx + du @ R @ du

    lx = grad(scalar_cost, argnums=0)(x, u)
    lu = grad(scalar_cost, argnums=1)(x, u)
    lxx = hessian(scalar_cost, argnums=0)(x, u)
    luu = hessian(scalar_cost, argnums=1)(x, u)

    lux = jacobian(grad(scalar_cost, argnums=1), argnums=0)(x, u)
    return lx, lu, lxx, luu, lux
  
def log_matrix(logger, name, matrix):
    """Универсальная функция логирования матрицы в ROS 2."""
    matrix_np = np.array(matrix)  # если это JAX, то .to_py() тоже может подойти
    logger.info(f"{name} shape: {matrix_np.shape}")
    logger.info(f"{name} contents:\n{matrix_np}")

@jit
def backward_pass(X, U, x_target_traj, u_target_traj, Q, R, Qf):
    Vx = grad(lambda x: jnp.dot((x - x_target_traj[-1]), Qf @ (x - x_target_traj[-1])))(X[-1])
    Vxx = Qf
    K_list = []
    k_list = []
    for k in reversed(range(horizon)):
        xk = X[k]
        uk = U[k]
        xt = x_target_traj[k]
        ut = u_target_traj[k]

        A, B = linearize_dynamics(xk, uk)

        # log_matrix(logger, "Matrix A", A)
        # log_matrix(logger, "Matrix B", B)
        # log_matrix(logger, "Matrix Vxx", Vxx)
        lx, lu, lxx, luu, lux = quadratize_cost(xk, uk, xt, ut, Q, R)

        # log_matrix(logger, "Matrix lux", lux)

        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx @ A
        Quu = luu + B.T @ Vxx @ B
        Qux = lux + B.T @ Vxx @ A #Qux = lux.T + B.T @ Vxx @ A  # обе части (m, n)

        Quu_reg = Quu + 1e-6 * jnp.eye(m)  # регуляризация
        Quu_inv = jnp.linalg.inv(Quu_reg)

        K = -Quu_inv @ Qux
        kff = -Quu_inv @ Qu

        Vx = Qx + K.T @ Quu @ kff + K.T @ Qu + Qux.T @ kff
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

        K_list.insert(0, K)
        k_list.insert(0, kff)
    return K_list, k_list

@jit
def forward_pass(X, U, k_list, K_list, alpha):
    x = X[0]
    X_new = [x]
    U_new = []
    for k in range(horizon):
        dx = x - X[k]
        du = k_list[k] + K_list[k] @ dx
        u_new = U[k] + alpha * du
        U_new.append(u_new)
        x = f(x, u_new, dt)
        X_new.append(x)
    return jnp.stack(X_new), jnp.stack(U_new)

@jit
def cost_function_traj_flip(x_traj, u_traj, x_target_traj, u_target_traj, Q, R, Qf):
    def compute_step_cost(x, u, x_target, u_target, is_terminal=False):
        # Вычисляем ошибку по позиции
        position_error = x[0:3] - x_target[0:3]
        position_cost = jnp.dot(position_error, jnp.dot(Q[0:3, 0:3], position_error))  # Матрица для позиции

        # Вычисляем ошибку по ориентации (кватернионы)
        q_current = x[6:10] / jnp.linalg.norm(x[6:10])  # Нормализация кватернионов
        q_target = x_target[6:10] / jnp.linalg.norm(x_target[6:10])
        dot_product = jnp.clip(jnp.dot(q_current, q_target), -1.0, 1.0)  # Ограничиваем скалярный продукт
        orientation_error = 2.0 * jnp.arccos(jnp.abs(dot_product))  # Ошибка по ориентации
        orientation_cost = orientation_error**2 * Q[6, 6]  # Штраф по элементам ориентации

        # Вычисляем ошибку по управлению
        control_error = u - u_target
        control_cost = jnp.dot(control_error, jnp.dot(R, control_error))  # Диагональная матрица для управления

        # Если это последний шаг (терминальный), то добавляем терминальный штраф
        if is_terminal:
            terminal_position_error = position_error
            terminal_orientation_error = orientation_error
            terminal_cost = jnp.dot(terminal_position_error, jnp.dot(Qf[0:3, 0:3], terminal_position_error)) + \
                            terminal_orientation_error**2 * Qf[6, 6]
            return position_cost + orientation_cost + control_cost + terminal_cost
            
        return position_cost + orientation_cost + control_cost

    # Суммируем стоимость по всем шагам траектории
    n_steps = x_traj.shape[0]
    step_costs = []

    for i in range(n_steps):
        is_terminal = (i == n_steps - 1)  # Последний шаг — терминальный
        step_cost = compute_step_cost(x_traj[i], u_traj[i], x_target_traj[i], u_target_traj[i], is_terminal)
        step_costs.append(step_cost)
        
    total_cost = jnp.sum(jnp.array(step_costs))
    return total_cost
 

def main(args=None):
    rclpy.init(args=args)
    node = ModelPredictiveControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

