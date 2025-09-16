#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.qos import   QoSProfile, ReliabilityPolicy,DurabilityPolicy, HistoryPolicy 
from px4_msgs.msg import (OffboardControlMode, VehicleThrustSetpoint, 
                            VehicleAttitudeSetpoint, TrajectorySetpoint, 
                            VehicleRatesSetpoint )
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
        self.thrust_setpoint_publisher = self.create_publisher(VehicleThrustSetpoint, "/fmu/in/vehicle_thrust_setpoint", qos_profile)
        self.attitude_setpoint_publisher = self.create_publisher(VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint", qos_profile)
        self.rates_setpoint_publisher = self.create_publisher(VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos_profile)
        # Подписка на разрешение
        self.control_permission = False
        self.create_subscription(Bool, 'control_permission', self.control_permission_callback, qos_profile)
        self.takeoff_height = -5.0
        # Логирование
        self.times = []
        self.thrusts = []

        self.start_time = time.time()
        self.get_logger().info("CustomController node started")

        self.lookahead_distance = 0.5        # расстояние "вперёд" по траектории для предсказания


        self.create_timer(0.1, self.position_control_loop)  # 20 Гц
        
        # Фигура Лиссажу с вычислением скорости
        def Lissajous_figure_func(self, t):
            A = 2.0
            B = 1.0
            omega = 0.2
            x = A * math.sin(omega * t)
            y = B * math.sin(omega * t) * math.cos(omega * t)   # y = (B/2) sin(2ω t)
            z = self.takeoff_height
            dx = A * omega * math.cos(omega * t)
            dy = B * omega * math.cos(2 * omega * t)
            return x, y, z, dx, dy

        def control_permission_callback(self, msg: Bool):
            self.control_permission = msg.data

        # Position setpoint
        def publish_position_setpoint(self, x, y, z):
            msg = TrajectorySetpoint()
            msg.position = [x, y, z]
            msg.yaw = 1.57079 # ?
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

        # Thrust setpoint
        def publish_thrust_setpoint(self, thrust_x, thrust_y, thrust_z):
            msg = VehicleThrustSetpoint()
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            msg.xyz = [thrust_x, thrust_y, thrust_z]
            self.thrust_setpoint_publisher.publish(msg)

        # Attitude setpoint
        def publish_attitude_setpoint(self, roll, pitch, yaw, thrust):
            msg = VehicleAttitudeSetpoint()
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            msg.roll_body = roll
            msg.pitch_body = pitch
            msg.yaw_body = yaw
            msg.thrust_body = [0.0, 0.0, -thrust]
            self.attitude_setpoint_publisher.publish(msg)

        # Rates setpoint
        def publish_rates_setpoint(self, roll_rate, pitch_rate, yaw_rate, thrust):
            msg = VehicleRatesSetpoint()
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            msg.roll = roll_rate
            msg.pitch = pitch_rate
            msg.yaw = yaw_rate
            msg.thrust_body = [0.0, 0.0, -thrust]
            self.rates_setpoint_publisher.publish(msg)

        # Control loops
        def position_control_loop(self):
            if self.control_permission:
                t = self.get_clock().now().nanoseconds / 1e9
                x, y, z, _, _ = self.Lissajous_figure_func(t)
                self.publish_position_setpoint(x, y, z)

        def thrust_control_loop(self):
            if self.control_permission:
                t = self.get_clock().now().nanoseconds / 1e9
                x, y, z, dx, dy = self.Lissajous_figure_func(t)
                # пример: просто берем скорость по x/y как thrust
                self.publish_thrust_setpoint(dx, dy, z)

        def attitude_control_loop(self):
            if self.control_permission:
                t = self.get_clock().now().nanoseconds / 1e9
                x, y, z, dx, dy = self.Lissajous_figure_func(t)
                g = 9.81
                roll = dy / g
                pitch = -dx / g
                yaw = 1.57
                thrust = 5.0
                self.publish_attitude_setpoint(roll, pitch, yaw, thrust)

        def rates_control_loop(self):
            if self.control_permission:
                t = self.get_clock().now().nanoseconds / 1e9
                x, y, z, dx, dy = self.Lissajous_figure_func(t)
                g = 9.81
                roll_rate = dy / g
                pitch_rate = -dx / g
                yaw_rate = 0.0
                thrust = 5.0
                self.publish_rates_setpoint(roll_rate, pitch_rate, yaw_rate, thrust)
        

    # нужно вычислить время полета однойфигуры
    # скорость
    
    # def roll_pith_yaw_contrtol_loop(self):
    #     dt = 0.01 # точнее
    #     t = self.get_clock().now().nanoseconds / 1e9  # текущее время в секундах
    #     # 1. Найти текущую позицию и скорость дрона
    #     x_cur, y_cur, z_cur = current_state.position
    #     vx_cur, vy_cur, vz_cur = current_state.velocity

    #     # 2. Посчитать текущую точку траектории
    #     x_traj, y_traj, z_traj = lissajous(t)
        
    #     # 3. Посчитать точку lookahead
    #     t_look = t + dt * lookahead_distance
    #     x_look, y_look, z_look = lissajous(t_look)

    #     # 4. Вектор направления (касательная к траектории)
    #     dx = x_look - x_cur
    #     dy = y_look - y_cur
    #     dz = z_look - z_cur
    #     distance = math.sqrt(dx**2 + dy**2 + dz**2)
    #     t_vec = (dx / distance, dy / distance, dz / distance)

    #     # 5. Вычисляем желаемое ускорение (feedforward)
    #     a_lat = distance / dt**2                  # грубое приближение бокового ускорения
    #     a_lat = min(a_lat, g * math.tan(phi_max)) # ограничение по физике дрона

    #     # 6. Определяем roll/pitch
    #     roll_cmd  = math.atan2(dy / dt, g)       # наклон в сторону поворота
    #     pitch_cmd = -math.atan2(dx / dt, g)      # наклон вперёд/назад
    #     roll_cmd  = max(-phi_max, min(phi_max, roll_cmd))   # ограничение
    #     pitch_cmd = max(-phi_max, min(phi_max, pitch_cmd)) # ограничение

    #     # 7. Yaw — направление вдоль траектории
    #     yaw_cmd = math.atan2(dy, dx)

    #     # 8. Тяга для поддержания высоты
    #     thrust = math.sqrt(g**2 + (dx/dt)**2 + (dy/dt)**2)  # грубое приближение

    #     # 9. Отправляем команды контроллеру
    #     send_to_drone(roll_cmd, pitch_cmd, yaw_cmd, thrust)

    
 


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
