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
        self.ki = 0.005
        self.kd = 0.1

        self.integral = 0.0
        self.prev_error = 0.0
        self.desired_distance = 1.0  # Desired distance from the wall (meters)
        self.lookahead_distance = 1.0  # Lookahead distance L (meters)
        self.tolerance = 0.05  # Tolerance for "going straight" (e.g., 5 cm)
        self.integral_limit = 1.0  # Limit for integral windup

    def get_range(self, range_data, angle_deg):
        """ 
        Get the LIDAR range at a specific angle in degrees.
        """
        angle_rad = np.radians(angle_deg)
        index = int((angle_rad - range_data.angle_min) / range_data.angle_increment)
        if index < 0 or index >= len(range_data.ranges):
            return None  # Ensure index is within bounds
        range_at_angle = range_data.ranges[index]
        if np.isinf(range_at_angle) or np.isnan(range_at_angle):
            return None
        return range_at_angle

    def calculate_alpha(self, a, b, theta_deg):
        """
        Calculate the angle alpha between the car's axis and the wall.
        """
        theta_rad = np.radians(theta_deg)
        
        # Add a small threshold to prevent division by near-zero values
        if abs(a * np.sin(theta_rad)) < 1e-6:
            return 0.0
        
        alpha_rad = np.arctan((a * np.cos(theta_rad) - b) / (a * np.sin(theta_rad)))

        return np.degrees(alpha_rad) + 45  # Return alpha in degrees


    def calculate_distance(self, b, alpha_deg):
        """
        Calculate the current distance to the wall (Dt).
        """
        alpha_rad = np.radians(alpha_deg)
        return b * np.cos(alpha_rad)

    def calculate_future_distance(self, D_t, alpha_deg):
        """
        Estimate the future distance to the wall (Dt+1).
        """
        alpha_rad = np.radians(alpha_deg)
        return D_t + self.lookahead_distance * np.sin(alpha_rad)

    def scan_callback(self, msg):
        """
        Callback function to process LIDAR scan data and compute the driving commands.
        """
        # Get two distances from LIDAR scans
        a = self.get_range(msg, 90)  # Distance at 90 degrees to the left
        b = self.get_range(msg, 45)  # Distance at 45 degrees to the left

        if a is not None and b is not None:
            theta_deg = 45  # Angle theta in degrees
            alpha_deg = self.calculate_alpha(a, b, theta_deg)  # Calculate alpha in degrees
            D_t = self.calculate_distance(b, alpha_deg)  # Calculate current distance Dt
            D_t_1 = self.calculate_future_distance(D_t, alpha_deg)  # Estimate future distance Dt+1

            # Calculate the error as the difference between the desired distance and D_t+1
            error = self.desired_distance - D_t_1

            # For now, we don't change the steering angle; we go straight.
            steering_angle = 0.0  # Fixed steering angle (go straight)

            # Adjust speed based on steering angle
            velocity = 1.0  # Keep a fixed speed for straight driving

            # Publish the steering angle and speed
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = velocity
            self.drive_publisher.publish(drive_msg)

            # Logs for debugging
            self.get_logger().info(f"Published Ackermann command -> Steering: {steering_angle:.4f}, Speed: {velocity:.4f}")
            self.get_logger().info(f"Range at 90 degrees: {a:.2f} meters")
            self.get_logger().info(f"Range at 45 degrees: {b:.2f} meters")
            self.get_logger().info(f"Alpha (angle to wall) calculated: {alpha_deg:.4f} degrees")
            self.get_logger().info(f"Current distance to wall (Dt): {D_t:.4f} meters")
            self.get_logger().info(f"Estimated future distance (Dt+1): {D_t_1:.4f} meters")
            self.get_logger().info(f"Error calculated: {error:.4f} meters")
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
