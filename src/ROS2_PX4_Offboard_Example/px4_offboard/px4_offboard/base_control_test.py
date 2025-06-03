import rclpy
from rclpy.node import Node
from px4_msgs.msg import ActuatorControls
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
class ActuatorControlsPublisher(Node):
    def __init__(self):
        super().__init__('actuator_controls_publisher')
        qos_profile = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10
                ) 
        self.publisher_ = self.create_publisher(
            ActuatorControls,
            '/fmu/in/actuator_controls_0',
            10
        )

        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Гц
        self.index = 0

    def timer_callback(self):
        msg = ActuatorControls()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # PX4 ожидает микросекунды

        # Пример: стабильные RPM по всем 4 моторам
        rpm_target = 1500  # Примерная цель (может потребоваться преобразование)
        # Преобразуй RPM в normalized [0.0, 1.0] throttle команду
        normalized = self.rpm_to_control(rpm_target)

        # Задаём значение тяги моторам
        msg.control = [normalized] * 4 + [0.0] * 4  # Всего 8 каналов

        # instance=0 соответствует main output group 0
        msg.actuator_group = 0
        msg.control_type = 0  # Обычно 0: control values (как PWM)

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published actuator_controls_0: {msg.control[:4]}')

    def rpm_to_control(self, rpm):
        # Псевдо-нормализация RPM в диапазон [0, 1]
        MAX_RPM = 2100.0  # как ты писала ранее
        return np.clip(rpm / MAX_RPM, 0.0, 1.0)

def main(args=None):
    rclpy.init(args=args)
    node = ActuatorControlsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()