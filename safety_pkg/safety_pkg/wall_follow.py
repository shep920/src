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

        # PID controller gains (adjusted)
        self.kp = 0.3
        self.ki = 0.002
        self.kd = 0.05

        self.integral = 0.0
        self.prev_error = 0.0
        self.desired_distance = 0.7  # Adjusted desired distance to 0.7 meters
        self.lookahead_distance = 0.5  # Increased lookahead distance to 0.5 meters
        self.tolerance = 0.1  # Increased tolerance to 10 cm
        self.integral_limit = 1.0  # Limit for integral windup

    def get_range(self, range_data, angle_deg):
        """ 
        Get the LIDAR range at a specific angle in degrees and filter out invalid data.
        """
        angle_rad = np.radians(angle_deg)
        index = int((angle_rad - range_data.angle_min) / range_data.angle_increment)
        if index < 0 or index >= len(range_data.ranges):
            return None  # Ensure index is within bounds
        range_at_angle = range_data.ranges[index]
        if np.isinf(range_at_angle) or np.isnan(range_at_angle) or range_at_angle > 10:
            return None  # Filter out invalid or large values (greater than 10m)
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
        return np.degrees(alpha_rad)  # Return alpha in degrees (removed +45 offset)

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

    def pid_control(self, error):
        """
        PID controller to compute the steering angle based on the error.
        """
        # Proportional term
        P = self.kp * error
        self.get_logger().info(f"Proportional (P): {P:.4f}")

        # Integral term (with clamping to avoid windup)
        self.integral += error
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)  # Clamp the integral term
        I = self.ki * self.integral
        self.get_logger().info(f"Integral (I): {I:.4f}")

        # Derivative term
        D = self.kd * (error - self.prev_error)
        self.get_logger().info(f"Derivative (D): {D:.4f}")

        # Calculate the steering angle
        steering_angle = P + I + D
        self.prev_error = error

        # Log the unclamped steering angle
        self.get_logger().info(f"Unclamped steering angle: {steering_angle:.4f}")

        # Limit the steering angle to a safe range (e.g., between -0.34 and 0.34 radians)
        steering_angle = max(min(steering_angle, 0.34), -0.34)
        self.get_logger().info(f"Clamped steering angle: {steering_angle:.4f}")

        return steering_angle

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

            # Use the PID controller to calculate the steering angle
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
