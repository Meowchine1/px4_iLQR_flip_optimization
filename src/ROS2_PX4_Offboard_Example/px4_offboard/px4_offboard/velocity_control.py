#!/usr/bin/env python


__author__ = "Meowchine1"
__contact__ = "meowchine111@gmail.com"

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy,DurabilityPolicy, HistoryPolicy 
import math
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude, VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleCommand
from geometry_msgs.msg import Twist, Vector3
from math import pi
from std_msgs.msg import Bool
import time

nav_state_dict = {
    VehicleStatus.NAVIGATION_STATE_MANUAL: "MANUAL",
    VehicleStatus.NAVIGATION_STATE_ALTCTL: "ALTCTL",
    VehicleStatus.NAVIGATION_STATE_POSCTL: "POSCTL",
    VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "AUTO_MISSION",
    VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "AUTO_LOITER",
    VehicleStatus.NAVIGATION_STATE_AUTO_RTL: "AUTO_RTL",
    VehicleStatus.NAVIGATION_STATE_POSITION_SLOW: "POSITION_SLOW",
    VehicleStatus.NAVIGATION_STATE_FREE5: "FREE5",
    VehicleStatus.NAVIGATION_STATE_FREE4: "FREE4",
    VehicleStatus.NAVIGATION_STATE_FREE3: "FREE3",
    VehicleStatus.NAVIGATION_STATE_ACRO: "ACRO",
    VehicleStatus.NAVIGATION_STATE_FREE2: "FREE2",
    VehicleStatus.NAVIGATION_STATE_DESCEND: "DESCEND",
    VehicleStatus.NAVIGATION_STATE_TERMINATION: "TERMINATION",
    VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
    VehicleStatus.NAVIGATION_STATE_STAB: "STABILIZED",
    VehicleStatus.NAVIGATION_STATE_FREE1: "FREE1",
    VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
    VehicleStatus.NAVIGATION_STATE_AUTO_LAND: "AUTO_LAND",
    VehicleStatus.NAVIGATION_STATE_AUTO_FOLLOW_TARGET: "AUTO_FOLLOW_TARGET",
    VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND: "AUTO_PRECLAND",
    VehicleStatus.NAVIGATION_STATE_ORBIT: "ORBIT",
    VehicleStatus.NAVIGATION_STATE_AUTO_VTOL_TAKEOFF: "AUTO_VTOL_TAKEOFF",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL1: "EXTERNAL1",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL2: "EXTERNAL2",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL3: "EXTERNAL3",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL4: "EXTERNAL4",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL5: "EXTERNAL5",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL6: "EXTERNAL6",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL7: "EXTERNAL7",
    VehicleStatus.NAVIGATION_STATE_EXTERNAL8: "EXTERNAL8",
    VehicleStatus.NAVIGATION_STATE_MAX: "MAX"
}

arm_state_dict = {
VehicleStatus.ARMING_STATE_DISARMED: "DISARMED",
VehicleStatus.ARMING_STATE_ARMED: "ARMED"
}

failsafe_dict = {
    0: "DISABLED",
    1: "ENABLED",
    2: "WOULD_FAILSAFE"
}
 

class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        self.offboard_velocity_sub = self.create_subscription(
            Twist,
            '/offboard_velocity_cmd',
            self.offboard_velocity_callback,
            qos_profile)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile) 

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_velocity = self.create_publisher(Twist, '/fmu/in/setpoint_velocity/cmd_vel_unstamped', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.publisher_attitude = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)

        #** connection between custom control and velocityu control
        self.control_permission_pub = self.create_publisher(Bool, 'control_permission', qos_profile)
        # Timers
        self.switch_to_ofboard_timer = self.create_timer(0.1, self.switch_to_ofboard_timer)
        self.vehicle_status_timer = self.create_timer(2., self.print_vehicle_status)

        # fileds initialization
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED 
        self.velocity = Vector3()
        self.yaw = 0.0  #yaw value we send as command
        self.trueYaw = 0.0  #current yaw value of drone
        self.truePitch = 0.0
        self.trueRoll = 0.0 
        self.flightCheck = False
        self.myCnt = 0
        self.offboard_state = False 
        self.failsafe = False
        self.current_state = "IDLE"
        self.offboard_setpoint_counter = 0
 

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def switch_to_ofboard_timer(self):
        self.publish_offboard_control_heartbeat_signal()
        
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            self.arm()

        if self.offboard_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.engage_offboard_mode()

        # if self.offboard_setpoint_counter == 10:
        #     self.engage_offboard_mode() 

        # if self.offboard_setpoint_counter < 11:
        #     self.offboard_setpoint_counter += 1

        #I tryed to implement state machine but it isnot work clearly as expected
        # match self.current_state:
        #     case "IDLE":
        #         if self.flightCheck:
        #             self.current_state = "ARMING"
        #             self.get_logger().info("move to ARMING state")
        #         #self.arm()

        #     case "ARMING":
        #         self.arm()
        #         if not self.flightCheck:
        #             self.current_state = "IDLE"
        #             #self.get_logger().info("ARMING --> back to IDLE state")
        #         elif self.arming_state == VehicleStatus.ARMING_STATE_ARMED and self.myCnt > 10:
        #             self.current_state = "TAKEOFF"
        #             #self.get_logger().info("move to TAKEOFF state")
                 

        #     case "TAKEOFF":
        #         self.take_off()
        #         if not self.flightCheck:
        #             self.current_state = "IDLE"
        #             #self.get_logger().info("TAKEOFF --> back to IDLE state")
        #         elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: 
        #             self.current_state = "OFFBOARD"
        #            # self.get_logger().info("move to OFFBOARD state")
                 
        #     case "OFFBOARD":
        #         self.state_offboard()
        #         if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
        #             self.current_state = "IDLE"
        #             #self.get_logger().info("Offboard --> back to IDLE state")
        #         elif self.offboard_state:
        #                 self.current_state = "OFFBOARD_FLYING"
        #                 #self.get_logger().info("Entered OFFBOARD_FLYING")
                 
        #     case "OFFBOARD_FLYING":
        #         # Проверка на потерю управления
        #         if not self.flightCheck:
        #             #self.get_logger().warn("OFFBOARD_FLYING --> back to IDLE state: flightCheck failed")
        #             self.current_state = "IDLE"
        #         elif self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
        #             #self.get_logger().warn("OFFBOARD_FLYING --> back to IDLE state: Disarmed")
        #             self.current_state = "IDLE"
        #         elif not self.offboard_state:
        #             #self.get_logger().warn("OFFBOARD_FLYING → back to OFFBOARD: Offboard lost")
        #             self.current_state = "OFFBOARD"
        #         elif self.failsafe:
        #             #self.get_logger().warn("OFFBOARD_FLYING --> back to IDLE state: Failsafe")
        #             self.current_state = "IDLE"
        # self.myCnt += 1

    def state_offboard(self):
        self.myCnt = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
 
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        #self.get_logger().info("Arm command send")
 
    def take_off(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1 = 1.0, param7=5.0) # param7 is altitude in meters
        #self.get_logger().info("Takeoff command send")
 
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
        self.vehicle_command_publisher_.publish(msg)
 

    def check_control_permission(self):
        msg = Bool()
        msg.data = (
            (self.arming_state == VehicleStatus.ARMING_STATE_ARMED) and self.offboard_state
        )
        self.control_permission_pub.publish(msg)

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
        self.failsafe = msg.failsafe
        self.flightCheck = msg.pre_flight_checks_pass
        # Обновление флага offboard
        self.offboard_state =  msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD 
        self.check_control_permission()

        #self.get_logger().info(" vehicle_status_callback")

    def print_vehicle_status(self):
        self.get_logger().info(
            f"\nNAV_STATUS:    {nav_state_dict.get(self.nav_state, 'UNKNOWN')}\n"
            f"ARM_STATUS:    {arm_state_dict.get(self.arming_state, 'UNKNOWN')}\n"
            f"FAILSAFE:      {failsafe_dict.get(self.failsafe)}\n"
            f"FlightCheck:   {self.flightCheck}\n"
            f"OffboardState: {self.offboard_state}\n"
            f"self.current_state {self.current_state}\n"
        )

    def offboard_velocity_callback(self, msg):
        self.velocity.x = -msg.linear.y
        self.velocity.y = msg.linear.x
        self.velocity.z = -msg.linear.z
        self.yaw = msg.angular.z
        self.roll = msg.angular.x
        self.pitch = msg.angular.y

    def attitude_callback(self, msg):
        x, y, z, w = msg.q
        # Формулы преобразования
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        self.trueYaw = - np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        self.truePitch = np.arcsin(t2)
        t3 = 2.0 * (w * x + y * z)
        t4 = 1.0 - 2.0 * (x * x + y * y)
        self.trueRoll = np.arctan2(t3, t4)
  

def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        offboard_control.get_logger().info(" KeyboardInterrupt .")
         
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
