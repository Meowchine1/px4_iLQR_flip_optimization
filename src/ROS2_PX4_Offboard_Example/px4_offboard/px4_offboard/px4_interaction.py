import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist, PoseStamped
from scipy.spatial.transform import Rotation as R 
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (  OffboardControlMode, TrajectorySetpoint, 
    VehicleStatus, VehicleRatesSetpoint, VehicleCommand,  
    VehicleAttitudeSetpoint, VehicleThrustSetpoint, VehicleTorqueSetpoint, ActuatorMotors
)
import numpy as np
from enum import Enum 
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import String
from rclpy.qos import QoSProfile
from quad_flip_msgs.msg import OptimizedTraj

from pymavlink import mavutil
import threading

from pymavlink.quaternion import QuaternionBase
from px4_msgs.msg import  EscStatus 
import math

BOUNCE_TIME = 0.6
ACCELERATE_TIME = 0.07
BRAKE_TIME = ACCELERATE_TIME
ARM_TIMEOUT = 5.0
OFFBOARD_TIMEOUT = 5.0
OFFBOARD_MODE = 14  # код режима OFFBOARD в PX4
class DroneState(Enum):
    INIT = 7
    DISARMED = 0
    ARMING = 1
    ARMED = 2
    OFFBOARD = 3
    TAKEOFF = 4
    FLIP = 6
    LANDING = 16
    MPC_MANAGEMENT = 15 

# dynamic drone control params
# TODO Move it to FILE
horizon = 50 # Горизонт предсказания
n = 13  # Размерность состояния квадрокоптера (позиция, скорость, ориентация, угловая скорость)
m = 4  # Размерность управления (4 мотора)
 

class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        """Инициализация PID-контроллера с заданными коэффициентами."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """
        Вычисляет управляющее воздействие на основе ошибки между заданным значением и измерением.
        
        :param setpoint: Желаемое значение (например, RPM)
        :param measurement: Текущее измеренное значение (например, RPM)
        :return: Управляющее воздействие
        """
        error = setpoint - measurement
        self.integral += error
        derivative = error - self.prev_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output

    def get_rotate_pwm(self, target_rpm, current_rpm):
        """
        Возвращает значение PWM на основе текущих и целевых RPM с использованием PID-регулятора.

        :param target_rpm: Целевое значение оборотов
        :param current_rpm: Текущее значение оборотов
        :return: Расчётное PWM значение в диапазоне [1000, 2000]
        """
        pid_output = self.compute(target_rpm, current_rpm)
        pwm = 1500.0 + pid_output
        return float(np.clip(pwm, 1000.0, 2000.0)) 

class FlipControlNode(Node):
    def __init__(self):
        super().__init__('flip_control_node')

        #qos_profile = QoSProfile(depth=10)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        ) 
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile) 
        self.vehicle_rates_publisher = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        
        self.attitude_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.motors_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
 
       # MPC INTEGRATION API
        self.pub_to_mpc = self.create_publisher(String, '/drone/client_msg', qos_profile)#
        self.create_subscription(OptimizedTraj, '/drone/optimized_traj', self.optimized_traj_callback, qos_profile)
        self.create_subscription(String, '/drone/server_msg', self.server_msg_callback, qos_profile)

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
         # ****** RPM *******
        self.create_subscription(EscStatus, '/fmu/out/esc_status', self.esc_status_callback, qos_profile)

        # == == == =STATE CONTROL= == == == 
        self.main_state = DroneState.INIT
        
        # == == == =PX4 STATES= == == == 
        self.arming_state = 0 
        self.nav_state = 0 
        self.vehicle_status = VehicleStatus()
        self.stage_time = time.time()
        self.offboard_is_active = False
        self.offboard_state = False 
        
        self.received_x_opt = np.zeros((horizon + 1, n))  # (N+1) x n
        self.received_u_opt = np.zeros((horizon, m))  # N x m
        self.received_i_final = 0
        self.received_cost_final = 0.0 
        self.received_done =  False
        self.target_u, self.target_x = [], []
        self.takeoff_alt = 0.0
        self.mpc_takeoff = False 
        self.drone_managenment_f = False
        self.rpm_to_pwm_pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.01)
        self.target_pwm = np.zeros(4, dtype=int)
        self.target_rpm = np.zeros(4, dtype=int)
        self.motor_rpms = np.zeros(4, dtype=int)
        self.max_rpm = 15000
 
        # #self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
        # self.arm()

        # msg = OffboardControlMode()
        # msg.position = True
        # msg.velocity = False
        # msg.acceleration = False
        # msg.attitude = False
        # msg.body_rate = False
        # msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # self.offboard_control_mode_publisher.publish(msg)


        # self.create_timer(0.1, self.update) 
        # self.create_timer(0.1, self.drone_managenment) 
        # self.create_timer(0.01, self.publish_all)  # 100 Hz

    def timestamp(self):
        return self.get_clock().now().nanoseconds // 1000  # микросекунды

    def publish_all(self):
        timestamp = self.timestamp()

        # # 1. VehicleAttitudeSetpoint
        attitude = VehicleAttitudeSetpoint()
        attitude.timestamp = timestamp
        attitude.q_d = [1.0, 0.0, 0.0, 0.0]  # уровень
        attitude.roll_body = 0.0
        attitude.pitch_body = 0.0
        attitude.yaw_body = 0.0
        attitude.yaw_sp_move_rate = 0.0
        attitude.thrust_body = [0.0, 0.0, -9.81]
        self.attitude_pub.publish(attitude)

        # 2. VehicleThrustSetpoint
        thrust = VehicleThrustSetpoint()
        thrust.timestamp = timestamp
        thrust.xyz = [0.0, 0.0, -9.81]
        self.thrust_pub.publish(thrust)

        # # 3. VehicleTorqueSetpoint
        # torque = VehicleTorqueSetpoint()
        # torque.timestamp = timestamp
        # torque.xyz = [0.0, 0.0, 0.0]
        # self.torque_pub.publish(torque)

        # # 4. ActuatorMotors
        # motors = ActuatorMotors()
        # motors.timestamp = timestamp
        # motors.control = [0.6, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.motors_pub.publish(motors)

    
    def send_pwm_loop(self):
        if not rclpy.ok():
            return

        level_quat = [1.0, 0.0, 0.0, 0.0]  # без наклона
        type_mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
        )
        # Убедись, что type_mask передаётся без изменений
        self.get_logger().info(f"type_mask: {type_mask}")  # Должно быть 0b00000111 = 7

        # Каждый вызов — отправляем команду с тягой 0.7, чтобы удерживать OFFBOARD
        self.master.mav.set_attitude_target_send(
            int(self.master.time_since('BOOT') * 1e3),
            self.master.target_system,
            self.master.target_component,
            type_mask,
            level_quat,
            0.0, 0.0, 0.0,
            0.7
        )
        #self.get_logger().info("send_pwm_loop")
   
    def rpm_to_pwm(self,rpm, rpm_min=0, rpm_max=10000, pwm_min=1000, pwm_max=2000):
        rpm = max(min(rpm, rpm_max), rpm_min)
        pwm = pwm_min + (rpm - rpm_min) * (pwm_max - pwm_min) / (rpm_max - rpm_min)
        return int(pwm)

    def esc_status_callback(self, msg: EscStatus):
        rpms = [esc.esc_rpm for esc in msg.esc[:msg.esc_count]]
        self.motor_rpms = np.array(rpms)
        #self.get_logger().info(f"self.motor_rpms: {self.motor_rpms}")

        pwms = [self.rpm_to_pwm(rpm) for rpm in self.motor_rpms]
        #self.get_logger().info(f"Estimated PWM: {pwms}")

    def get_pwm(self):
        self.target_pwm = [
            int(self.rpm_to_pwm_pid.get_rotate_pwm(self.target_rpm[i], self.motor_rpms[i]))
            for i in range(4)
        ]
        self.get_logger().info(f'self.target_rpm: {self.target_rpm}')
      
    def drone_managenment(self):
        if self.drone_managenment_f:
            self.send_motor_commands()
         
    def optimized_traj_callback(self, msg: OptimizedTraj):
        self.received_x_opt = np.array(msg.x_opt, dtype=np.float32)
        self.received_u_opt = np.array(msg.u_opt, dtype=np.float32)
        self.target_rpm = np.array(msg.u_opt, dtype=np.float32)
        self.get_pwm()
        self.received_i_final = msg.i_final
        self.received_cost_final = msg.cost_final
        self.received_done = msg.done
        self.get_logger().info(f'px4 OptimizedTraj: {msg}') 

    def send_message_to_server(self, msg):#
        ros_msg = String()
        ros_msg.data = msg
        self.pub_to_mpc.publish(ros_msg)
        #self.get_logger().info(f'Sent to MPC: {msg}')

    def server_msg_callback(self, msg):
        data = msg.data
        #self.get_logger().info(f'server_msg_callback: {data} self.drone_managenment_f= {self.drone_managenment_f}')
        if data =='land':
            self.main_state == DroneState.LANDING
            self.drone_managenment_f = False
        elif data =='mpc_on':
            self.main_state = DroneState.MPC_MANAGEMENT
            self.drone_managenment_f = True 
        
        # NOT WORKING 
    def vehicle_status_callback(self, msg):
        """Обновляет состояние дрона."""
        self.get_logger().info('vehicle_status_callback')
        self.vehicle_status = msg
        self.arming_state = msg.arming_state
        
        if msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.offboard_state = True
        else:
            self.offboard_state = False
            self.get_logger().info(f"Текущий режим: {msg.nav_state}")
    
    def rpm_to_normalized(self, rpms):
        return [(rpm / self.max_rpm) ** 2 for rpm in rpms]
# MOTOR MANAGEMENT
    def send_motor_commands(self):
        # нормализованные значения [0,1] — переводим в pwm/thrust команды
        motor_inputs = self.rpm_to_normalized(self.target_rpm)  # типичный размер: 4
        motor_array = np.clip(motor_inputs, 0.0, 1.0).astype(float).tolist()

        # Дополняем до 12 значений нулями
        if len(motor_array) < 12:
            motor_array += [0.0] * (12 - len(motor_array))
        elif len(motor_array) > 12:
            motor_array = motor_array[:12]  # обрежем лишнее, если что-то пошло не так

        # Создаём и публикуем сообщение
        msg = ActuatorMotors()
        msg.control = motor_array
        self.publisher_actuator_motors.publish(msg)

        self.get_logger().info(f'Sent motor_commands: {motor_array}')

    def reset_rate_pid(self): 
        self.publish_rate_setpoint(roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0)
        self.get_logger().info(f'reset_rate_pid')
    def set_thrust(self, thrust):
            msg = VehicleAttitudeSetpoint()
            msg.thrust_body[2] = -thrust
            self.publisher_att.publish(msg)
# COMMANDS
    def send_takeoff_commanad(self, altitude: float):
        takeoff_cmd = VehicleCommand()
        takeoff_cmd.command = VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF

        takeoff_cmd.param7 = altitude  # Целевая абсолютная высота (в метрах)

        # Остальные параметры можно оставить по умолчанию
        takeoff_cmd.param1 = 0.0  # Минимальная высота (не используется)
        takeoff_cmd.param2 = 0.0  # Прецизионный режим
        takeoff_cmd.param3 = 0.0  # пусто
        takeoff_cmd.param4 = float('nan')  # yaw
        takeoff_cmd.param5 = float('nan')  # latitude
        takeoff_cmd.param6 = float('nan')  # longitude

        takeoff_cmd.target_system = 1
        takeoff_cmd.target_component = 1
        takeoff_cmd.source_system = 1
        takeoff_cmd.source_component = 1
        takeoff_cmd.from_external = True

        self.vehicle_command_publisher.publish(takeoff_cmd)
        self.get_logger().info(f'Sending takeoff command to altitude {altitude:.2f} m.')

    def send_land_command(self):
        land_cmd = VehicleCommand()
        land_cmd.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
        # param1: Abort alt (0 = disable)
        land_cmd.param1 = 0.0
        # param2: Precision land mode (0 = disabled, 1 = enabled)
        land_cmd.param2 = 0.0
        # param3: Empty
        land_cmd.param3 = float('nan')
        # param4: Desired yaw angle (NaN = keep current)
        land_cmd.param4 = float('nan')
        # param5, param6: latitude, longitude (NaN = current position)
        land_cmd.param5 = float('nan')
        land_cmd.param6 = float('nan')
        # param7: Altitude (NaN = current position)
        land_cmd.param7 = float('nan')

        self.vehicle_command_publisher.publish(land_cmd)
        self.get_logger().info('Sending land command.')
    
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Отправка команды дрону."""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(time.time() * 1e6)
        self.vehicle_command_publisher.publish(msg)
        #self.get_logger().info(f'publish_vehicle_command')

# ВКЛЮЧИТЬ РЕЖИМЫ
    def set_stabilization_mode(self):
        """Переводит дрон в режим стабилизации (STABILIZE)."""
        msg = VehicleCommand()
        msg.command = 1  # Команда для перехода в режим стабилизации (STABILIZE)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(time.time() * 1e6)
        self.vehicle_command_publisher.publish(msg)
        self.get_logger().info(f'set_stabilization_mode')
     
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    """ Дрон должен постоянно получать это сообщение чтобы оставаться в offboard """
    def offboard_heartbeat(self):
         if self.offboard_is_active:
                #self.get_logger().info("Sending SET_MODE OFFBOARD") 
                """Publish the offboard control mode."""
                msg = OffboardControlMode()
                msg.position = True
                msg.velocity = False
                msg.acceleration = False
                msg.attitude = False
                msg.body_rate = False
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.offboard_control_mode_publisher.publish(msg)
    def set_offboard_mode(self):
            """Switch to offboard mode."""
            self.offboard_is_active = True 
    """Управление на моторы нужно подавать непрерывно и с частотой не мене 0.01 сек"""
    def flip_thrust_max(self):
        if self.flip_thrust_max_f:
            self.set_thrust(1.0)
    def flip_thrust_recovery(self):
        if self.flip_thrust_recovery_f:
            self.set_thrust(0.6)
    def flip_pitch_t(self):
        if self.flip_pitch_f:
            #self.set_rates(17.0, 0.0, 0.0, 0.25)# roll_max_rate should be 1000 in QGC vechicle setup
            self.set_rates(25.0, 0.0, 0.0, 0.25)

     
    # main spinned function
    def update(self):
        #self.get_logger().info(f"self.main_state={self.main_state}  self.flip_state={self.flip_state}")
        #self.get_logger().info(f"UPDATE self.alt={self.alt}  self.vehicle_local_position.z={self.vehicle_local_position.z}")
        if self.main_state == DroneState.INIT:
            # self.set_offboard_mode()
             
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                pass#self.arm()
                return
            self.main_state = DroneState.ARMED 
         
        elif self.main_state == DroneState.ARMED:
             if self.offboard_state:
                 self.send_message_to_server("takeoff")#
            #pass
            
            #self.send_takeoff_commanad(5.0)

        # elif self.main_state == DroneState.MPC_MANAGEMENT:
             
        elif self.main_state == DroneState.LANDING:
            self.send_land_command()

def main(args=None):
    rclpy.init(args=args)
    node = FlipControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
