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
from rclpy.clock import Clock

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
 
nav_state_dict = {
    VehicleStatus.NAVIGATION_STATE_MANUAL: "MANUAL",
    VehicleStatus.NAVIGATION_STATE_ALTCTL: "ALTCTL",
    VehicleStatus.NAVIGATION_STATE_POSCTL: "POSCTL",
    VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "AUTO_MISSION",
    VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "AUTO_LOITER",
    VehicleStatus.NAVIGATION_STATE_AUTO_RTL: "AUTO_RTL",
    VehicleStatus.NAVIGATION_STATE_ACRO: "ACRO",
    VehicleStatus.NAVIGATION_STATE_DESCEND: "DESCEND",
    VehicleStatus.NAVIGATION_STATE_TERMINATION: "TERMINATION",
    VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
    VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
    VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND: "AUTO_PRECLAND",
    VehicleStatus.NAVIGATION_STATE_AUTO_LAND: "AUTO_LAND",
}
arming_state_dict = {
    VehicleStatus.ARMING_STATE_DISARMED: "DISARMED",
    VehicleStatus.ARMING_STATE_ARMED: "ARMED",
}

class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:

        error = setpoint - measurement
        self.integral += error
        derivative = error - self.prev_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output

    def get_rotate_pwm(self, target_rpm, current_rpm):
     
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
        # Drone movement topics
        self.attitude_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.motors_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)

        self.pub_to_mpc = self.create_publisher(String, '/drone/client_msg', qos_profile)
        
        self.create_subscription(
            OptimizedTraj, 
            '/drone/optimized_traj', 
            self.optimized_traj_callback, 
            qos_profile) 
        self.create_subscription(
            String, 
            '/drone/server_msg', 
            self.server_msg_callback, 
            qos_profile) 
        self.create_subscription(
            VehicleStatus, 
            '/fmu/out/vehicle_status', 
            self.vehicle_status_callback, 
            qos_profile)
        # get actual RPM
        self.create_subscription(
            EscStatus, 
            '/fmu/out/esc_status', 
            self.esc_status_callback, 
            qos_profile)
        
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.velocity = Vector3()
        self.yaw = 0.0  #yaw value we send as command
        self.trueYaw = 0.0  #current yaw value of drone
        self.truePitch = 0.0
        self.trueRoll = 0.0
        self.offboardMode = False
        self.flightCheck = False
        self.myCnt = 0
        self.arm_message = False
        self.failsafe = False
        self.current_state = "IDLE"
        self.last_state = self.current_state 
        # == == == =STATE CONTROL= == == == 
        self.main_state = DroneState.INIT 
        # == == == =PX4 STATES= == == ==  
        self.stage_time = time.time() 
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
 
        arm_timer_period = .1 # seconds
        self.arm_timer_ = self.create_timer(arm_timer_period, self.arm_timer_callback)
 
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback) 

    def arm_timer_callback(self): 
        match self.current_state:
            case "IDLE":
                if self.flightCheck and self.arm_message:
                    self.current_state = "ARMING"
                    self.get_logger().info("Arming")
                    self.myCnt = 0  # сброс счётчика при входе

            case "ARMING":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Arming, Flight Check Failed")
                elif self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self.myCnt > 10:
                    self.current_state = "TAKEOFF"
                    self.get_logger().info("Arming, Takeoff")
                    self.myCnt = 0
                else:
                    # Отправляем arm-команду, если ещё не заармлены
                    if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
                        self.arm()

            case "TAKEOFF":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Takeoff, Flight Check Failed")
                elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self.current_state = "LOITER"
                    self.get_logger().info("Takeoff, Loiter")
                    self.set_mode_loiter()
                    self.myCnt = 0
                else:
                    # Отправляем команду takeoff (предполагается, что режим установлен в AUTO.TAKEOFF)
                    self.take_off()

            case "LOITER":
                if not self.flightCheck:
                    self.current_state = "IDLE"
                    self.get_logger().info("Loiter, Flight Check Failed")
                elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self.current_state = "OFFBOARD"
                    self.get_logger().info("Loiter, Offboard")
                    self.myCnt = 0
                else:
                    # Здесь можно держать LOITER, режим уже выставлен
                    pass

            case "OFFBOARD":
                if (not self.flightCheck or
                    self.arm_state != VehicleStatus.ARMING_STATE_ARMED or
                    self.failsafe):
                    self.current_state = "IDLE"
                    self.get_logger().info("Offboard, Flight Check Failed")
                else:
                    self.state_offboard()

        # Сбрасываем arm_message, если не заармлены
        if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
            self.arm_message = False

        # Логируем изменение состояния
        if self.last_state != self.current_state:
            self.last_state = self.current_state
            self.get_logger().info(f"State changed to {self.current_state}")

        self.myCnt += 1


    def state_offboard(self):
        self.myCnt = 0
        # Вызов режима OFFBOARD (6 — это VehicleCommand.MODE_OFFBOARD)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
        self.offboardMode = True
 
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")
 
    def take_off(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1=1.0, param7=5.0)
        self.get_logger().info("Takeoff command sent")
 
    def set_mode_loiter(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # custom mode
            param2=float(VehicleCommand.MODE_AUTO),
            param3=float(VehicleCommand.SUB_MODE_AUTO_LOITER)
        )
        self.get_logger().info("Mode set to AUTO.LOITER")
 
#publishes offboard control modes and velocity as trajectory setpoints
    def cmdloop_callback(self):
        if(self.offboardMode == True):
            # Publish offboard control modes
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            offboard_msg.position = False
            offboard_msg.velocity = False
            offboard_msg.acceleration = False
            self.publisher_offboard_mode.publish(offboard_msg)            


    def vehicle_status_callback(self, msg):
        # Логирование навигационного состояния, если изменилось
        if msg.nav_state != self.nav_state:
            nav_str = nav_state_dict.get(msg.nav_state, f"UNKNOWN({msg.nav_state})")
            self.get_logger().info(f"NAV_STATUS: {nav_str}")
        
        # Логирование статуса армирования, если изменилось
        if msg.arming_state != self.arming_state:
            arm_str = arming_state_dict.get(msg.arming_state, f"UNKNOWN({msg.arming_state})")
            self.get_logger().info(f"ARM STATUS: {arm_str}")

        # Логирование failsafe
        if msg.failsafe != self.failsafe:
            self.get_logger().info(f"FAILSAFE: {msg.failsafe}")

        # Логирование результата предполетных проверок
        if msg.pre_flight_checks_pass != self.flightCheck:
            self.get_logger().info(f"FlightCheck: {msg.pre_flight_checks_pass}")

        # Обновление внутренних переменных
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
        self.failsafe = msg.failsafe
        self.flightCheck = msg.pre_flight_checks_pass

        # Обновление флага offboard
        self.offboard_state = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

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

        # 3. VehicleTorqueSetpoint
        torque = VehicleTorqueSetpoint()
        torque.timestamp = timestamp
        torque.xyz = [0.0, 0.0, 0.0]
        self.torque_pub.publish(torque)

        # 4. ActuatorMotors
        motors = ActuatorMotors()
        motors.timestamp = timestamp
        motors.control = [0.6, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motors_pub.publish(motors)

    
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
        #self.get_logger().info(f'Send to optimizer: {msg}')

    def server_msg_callback(self, msg):
        data = msg.data
        #self.get_logger().info(f'server_msg_callback: {data} self.drone_managenment_f= {self.drone_managenment_f}')
        if data =='land':
            self.main_state == DroneState.LANDING
            self.drone_managenment_f = False
        elif data =='mpc_on':
            self.main_state = DroneState.MPC_MANAGEMENT
            self.drone_managenment_f = True 
        

    def rpm_to_normalized(self, rpms):
        return [(rpm / self.max_rpm) ** 2 for rpm in rpms]

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
    
    #publishes command to /fmu/in/vehicle_command
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7    # altitude value in takeoff command
        msg.command = command  # command ID
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher.publish(msg)

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

     

def main(args=None):
    rclpy.init(args=args)
    node = FlipControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
