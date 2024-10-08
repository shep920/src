#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        
        # Subscribe to LaserScan for distance to obstacles
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Subscribe to Odometry to get current speed
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )

        # Publisher to publish the braking message if needed
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        self.speed = 0.0  # Initialize speed
        self.get_logger().info('SafetyNode has been started.')

    def odom_callback(self, odom_msg):
        # Update speed from the Odometry message and print the speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # Track the minimum iTTC
        min_ttc = float('inf')  # Initialize to a high value
        range_threshold = 0.5  # Distance threshold (1 meter)

        # Track the minimum range (distance to the nearest obstacle)
        min_range = min(scan_msg.ranges)

        # Brake if the car is closer than 1 meter to any obstacle
        if min_range < range_threshold:
            self.get_logger().info(f'Braking: Obstacle {min_range:.2f} meters away')
            self.brake()
            return  # Stop further processing since we're braking

        # Loop through each range in the LaserScan message for iTTC calculation
        for i, range_val in enumerate(scan_msg.ranges):
            if np.isinf(range_val) or np.isnan(range_val):
                continue  # Skip invalid range values

            # Calculate the angle of the beam
            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            # Calculate the range rate using the car's speed and the angle of the scan beam
            range_rate = -self.speed * np.cos(angle)

            # Calculate iTTC if the range rate is negative (i.e., moving toward an obstacle)
            if range_rate < 0:
                ttc = range_val / abs(range_rate)
                min_ttc = min(min_ttc, ttc)  # Track the minimum iTTC

        # Check if the minimum iTTC is below a threshold, brake if necessary
        if min_ttc < 0.5:  # Example threshold for braking based on iTTC
            self.brake()

    def brake(self):
        # Create an AckermannDriveStamped message to stop the car
        brake_msg = AckermannDriveStamped()
        brake_msg.drive.speed = 0.0
        self.drive_publisher.publish(brake_msg)
        self.get_logger().info('Emergency braking!')

    def brake(self):
        # Create an AckermannDriveStamped message to stop the car
        brake_msg = AckermannDriveStamped()
        brake_msg.drive.speed = 0.0
        self.drive_publisher.publish(brake_msg)
        self.get_logger().info('Emergency braking!')


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
