#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        # PID controller gains
        self.kp = 0.5
        self.ki = 0.01
        self.kd = 0.1

        self.integral = 0.0
        self.prev_error = 0.0
        self.desired_distance = 0.5  # Desired distance from the wall (meters)
        self.lookahead_distance = 1.0  # Lookahead distance L (meters)
        self.tolerance = 0.05  # Tolerance for "going straight" (e.g., 5 cm)

    def get_range(self, range_data, angle):
        """ 
        Get the LIDAR range at a specific angle in degrees.
        """
        angle = np.radians(angle)
        index = int((angle - range_data.angle_min) / range_data.angle_increment)
        range_at_angle = range_data.ranges[index]
        if np.isinf(range_at_angle) or np.isnan(range_at_angle):
            return None
        return range_at_angle

    def calculate_alpha(self, a, b, theta):
        """
        Calculate the angle alpha between the car's axis and the wall.
        """
        alpha = np.arctan((a * np.cos(theta) - b) / (a * np.sin(theta)))
        return alpha

    def calculate_distance(self, b, alpha):
        """
        Calculate the current distance to the wall (Dt).
        """
        return b * np.cos(alpha)

    def calculate_future_distance(self, D_t, alpha):
        """
        Estimate the future distance to the wall (Dt+1).
        """
        return D_t + self.lookahead_distance * np.sin(alpha)

    def pid_control(self, error):
        """
        PID controller to compute the steering angle based on the error.
        """
        P = self.kp * error
        self.integral += error
        I = self.ki * self.integral
        D = self.kd * (error - self.prev_error)
        steering_angle = P + I + D
        self.prev_error = error

        # Limit the steering angle to a safe range (e.g., between -0.34 and 0.34 radians)
        steering_angle = max(min(steering_angle, 0.34), -0.34)
        return steering_angle

    def scan_callback(self, msg):
        """
        Callback function to process LIDAR scan data and compute the driving commands.
        """
        # Get two distances from LIDAR scans
        a = self.get_range(msg, 90)  # Distance at 90 degrees to the right
        b = self.get_range(msg, 45)  # Distance at 45 degrees to the right

        if a is not None and b is not None:
            theta = np.radians(45)  # Angle theta in radians
            alpha = self.calculate_alpha(a, b, theta)  # Calculate alpha
            D_t = self.calculate_distance(b, alpha)  # Calculate current distance Dt
            D_t_1 = self.calculate_future_distance(D_t, alpha)  # Estimate future distance Dt+1

            # Calculate the error as the difference between the desired distance and D_t+1
            error = self.desired_distance - D_t_1

            # Only apply steering if the error is outside the tolerance range
            if abs(error) < self.tolerance:
                steering_angle = 0.0  # Go straight if within tolerance
            else:
                steering_angle = self.pid_control(error)

            # Adjust speed based on the magnitude of the steering angle
            if abs(steering_angle) < np.radians(10):
                velocity = 1.5  # High speed for straight driving
            elif abs(steering_angle) < np.radians(20):
                velocity = 1.0  # Moderate speed for slight turns
            else:
                velocity = 0.5  # Slow speed for sharp turns

            # Publish the steering angle and speed
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = velocity
            self.drive_publisher.publish(drive_msg)

            self.get_logger().info(f"Steering: {steering_angle:.2f}, Speed: {velocity:.2f}, Error: {error:.2f}")
        else:
            self.get_logger().warn("Invalid LIDAR data. Could not compute a or b.")

def main(args=None):
    rclpy.init(args=args)
    wall_follow_node = WallFollowNode()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
