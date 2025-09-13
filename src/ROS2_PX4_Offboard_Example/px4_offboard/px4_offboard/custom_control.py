#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.qos import   QoSProfile, ReliabilityPolicy,DurabilityPolicy, HistoryPolicy 
from px4_msgs.msg import OffboardControlMode, VehicleThrustSetpoint, VehicleAttitudeSetpoint, TrajectorySetpoint
import matplotlib.pyplot as plt
import math
class CustomController(Node):
    def __init__(self):
        super().__init__('custom_controller')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Паблишеры
        self.offb_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.attitude_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        # Подписка на разрешение
        self.control_permission = False
        self.create_subscription(Bool, 'control_permission', self.control_permission_callback, qos_profile)
        self.takeoff_height = -5.0
        # Логирование
        self.times = []
        self.thrusts = []

        self.start_time = time.time()
        self.get_logger().info("CustomController node started")

        self.create_timer(0.1, self.control_loop)  # 20 Гц

    def control_permission_callback(self, msg: Bool):
        self.control_permission = msg.data

    def publish_position_setpoint(self, x: float, y: float, z: float):
            """Publish the trajectory setpoint."""
            msg = TrajectorySetpoint()
            msg.position = [x, y, z]
            msg.yaw = 1.57079  # (90 degree)
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)
            self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def control_loop(self):
        if self.control_permission:
            t = self.get_clock().now().nanoseconds / 1e9  # текущее время в секундах
            A = 2.0  # амплитуда по X
            B = 1.0  # амплитуда по Y
            omega = 0.2  # частота колебаний

            # Траектория в виде лемнискаты (x = A * sin(wt), y = B * sin(wt) * cos(wt))
            x = A * math.sin(omega * t)
            y = B * math.sin(omega * t) * math.cos(omega * t)
            z = self.takeoff_height  # удерживаем заданную высоту

            self.publish_position_setpoint(x, y, z)

 

    def plot_data(self):
        plt.figure()
        plt.plot(self.times, self.thrusts, label='thrust_z')
        plt.xlabel('Time (s)')
        plt.ylabel('Thrust (NED Z)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('thrust_plot.png')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = CustomController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
