import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist, PoseStamped
from scipy.spatial.transform import Rotation as R 
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    VehicleAngularVelocity, VehicleAngularAccelerationSetpoint,  
    SensorCombined, VehicleImu, VehicleOdometry, OffboardControlMode, TrajectorySetpoint, 
    VehicleStatus, VehicleLocalPosition, 
      VehicleAttitude,
    VehicleRatesSetpoint, VehicleTorqueSetpoint, VehicleAttitudeSetpoint, VehicleCommand, ActuatorMotors
)
import numpy as np
from enum import Enum
from px4_offboard.mpc_controller import MPCController
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry  # inner ROS2 SEKF
from std_msgs.msg import String
from rclpy.qos import QoSProfile
from quad_flip_msgs.msg import OptimizedTraj

BOUNCE_TIME = 0.6
ACCELERATE_TIME = 0.07
BRAKE_TIME = ACCELERATE_TIME
ARM_TIMEOUT = 5.0
OFFBOARD_TIMEOUT = 5.0
 
class DroneState(Enum):
    INIT = 7
    DISARMED = 0
    ARMING = 1
    ARMED = 2
    OFFBOARD = 3
    TAKEOFF = 4
    FLIP = 6
    LANDING = 16

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
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_rates_publisher = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.vehicle_torque_publisher = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.publisher_rates = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.publisher_att = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
  
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
       
        # == == == =STATE CONTROL= == == == 
        self.main_state = DroneState.INIT
        
        # == == == =PX4 STATES= == == == 
        self.arming_state = 0 
        self.nav_state = 0 
        self.vehicle_status = VehicleStatus() 
        self.stage_time = time.time()
        self.offboard_is_active = False 
         
        self.create_timer(0.1, self.update)
        self.create_timer(0.1, self.offboard_heartbeat)
        self.create_timer(0.01, self.flip_thrust_max) 
        self.create_timer(0.001, self.flip_thrust_recovery) 
        self.create_timer(0.0001, self.flip_pitch_t)
        # == == == =Timer flags == == == =
        self.flip_thrust_max_f = False
        self.flip_thrust_recovery_f = False
        self.flip_pitch_f = False

        # MPC INTEGRATION API
        self.pub_to_mpc = self.create_publisher(String, '/drone/client_msg', qos_profile)#
        self.create_subscription(OptimizedTraj, '/drone/optimized_traj', self.optimized_traj_callback, qos_profile)
        self.create_subscription(String, '/drone/server_msg', self.server_msg_callback, qos_profile)
         
        # TODO FROOM FILE
        self.horizon = 50  # Горизонт предсказания
        self.n = 13  # Размерность состояния квадрокоптера (позиция, скорость, ориентация, угловая скорость)
        self.m = 4  # Размерность управления (4 мотора)

        self.received_x_opt = None
        self.received_u_opt = None
        self.received_i_final = None
        self.received_cost_final = None
        self.received_done = None

        self.target_u, self.target_x = [], []
         
        self.takeoff_alt = 0.0
        self.mpc_takeoff = False 
 
    def optimized_traj_callback(self, msg: OptimizedTraj):
        self.received_x_opt = np.array(msg.x_opt, dtype=np.float32)
        self.received_u_opt = np.array(msg.u_opt, dtype=np.float32)
        self.received_i_final = msg.i_final
        self.received_cost_final = msg.cost_final
        self.received_done = msg.done

        self.get_logger().info(f'OptimizedTraj: {msg}')

    def send_message_to_server(self, msg):#
        ros_msg = String()
        ros_msg.data = msg
        self.pub_to_mpc.publish(ros_msg)
        #self.get_logger().info(f'Sent to MPC: {msg}')

    def server_msg_callback(self, msg):
        self.get_logger().info(f'server_msg_callback: {msg}')
        if msg =="land":
            self.main_state == DroneState.LANDING
        elif msg =="flip":
            self.main_state = DroneState.FLIP

    def vehicle_status_callback(self, msg): 
        """Обновляет состояние дрона."""
        #self.get_logger().info('vehicle_status_callback')
        self.vehicle_status = msg
        self.arming_state = msg.arming_state
    
# MOTOR MANAGEMENT
    def send_motor_commands(self):
        # нормализованные значения [0,1] — переводим в pwm/thrust команды
        motor_inputs = self.received_u_opt ## обратная динамика
        msg = ActuatorMotors()
        msg.control = list(np.clip(motor_inputs, 0.0, 1.0))
        self.motor_pub.publish(msg)


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

    def publish_position_setpoint(self, x, y, z):
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.position = [x, y, -z]
        trajectory_msg.timestamp = int(time.time() * 1e6)
        self.trajectory_setpoint_publisher.publish(trajectory_msg) 
 
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
    def set_offboard_mode(self):
        """Switch to offboard mode."""
        self.offboard_is_active = True
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
            self.set_offboard_mode()
            self.arm()
            self.main_state = DroneState.ARMING 
         
        elif self.main_state == DroneState.ARMING:
            
            #self.get_logger().warn(f"arm_state: {self.arming_state}")
            #self.get_logger().info(f"nav_state: {self.nav_state}")
            if self.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                #self.get_logger().info('ARMING_STATE_ARMED')
                #self.main_state = DroneState.TAKEOFF
                self.send_message_to_server("takeoff")#
            else:
                self.set_offboard_mode()
                self.arm()
        # elif self.main_state == DroneState.TAKEOFF:
             
        #     if self.mpc_takeoff:
        #         self.main_state = DroneState.FLIP
   
        # elif self.main_state == DroneState.FLIP: 
        #     #optimal_motor_inputs = self.get_control_from_mpc()  # Вычисление оптимального управляющего воздействия
        #     #self.send_motor_commands(optimal_motor_inputs)

            
                
        # elif self.main_state == DroneState.LANDING:
        #     self.send_land_command()

def main(args=None):
    rclpy.init(args=args)
    node = FlipControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
