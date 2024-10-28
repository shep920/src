#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class FollowTheGap(Node):
    def __init__(self):
        super().__init__('follow_the_gap')
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        self.safety_radius = 0.5  # Safety bubble radius (meters)
        self.straight_speed = 1.5
        self.turn_speed = 0.5
        self.max_steering_angle = 0.34  # Max steering angle (radians)

    def scan_callback(self, msg):
        # Step 1: Preprocess laser scan data
        ranges = np.array(msg.ranges)
        ranges = np.where((ranges == 0) | (ranges > msg.range_max), np.inf, ranges)

        # Step 2: Find the closest point
        closest_idx = np.argmin(ranges)
        closest_dist = ranges[closest_idx]
        self.get_logger().info(f"Closest distance: {closest_dist:.2f} meters")

        # Step 3: Apply a safety bubble
        bubble_radius = int(np.degrees(self.safety_radius / msg.angle_increment))
        start_idx = max(0, closest_idx - bubble_radius)
        end_idx = min(len(ranges), closest_idx + bubble_radius)
        ranges[start_idx:end_idx] = np.inf  # Set ranges within the bubble to inf

        # Step 4: Find the largest gap
        gaps = self.find_gaps(ranges)
        if not gaps:
            self.get_logger().warn("No gaps found!")
            self.publish_drive_command(0.0, 0.0)
            return

        # Step 5: Find the best point in the largest gap
        largest_gap = max(gaps, key=lambda x: x[1] - x[0])
        best_idx = (largest_gap[0] + largest_gap[1]) // 2
        goal_angle = msg.angle_min + best_idx * msg.angle_increment

        # Step 6: Drive towards the goal point
        steering_angle = max(min(goal_angle, self.max_steering_angle), -self.max_steering_angle)
        speed = self.turn_speed if abs(steering_angle) > 0.1 else self.straight_speed
        self.publish_drive_command(steering_angle, speed)

    def find_gaps(self, ranges):
        """
        Finds and returns a list of gaps as tuples of (start_idx, end_idx).
        """
        gaps = []
        start_idx = None
        for i in range(1, len(ranges)):
            if np.isfinite(ranges[i]) and not np.isfinite(ranges[i - 1]):
                start_idx = i
            elif not np.isfinite(ranges[i]) and np.isfinite(ranges[i - 1]) and start_idx is not None:
                gaps.append((start_idx, i))
                start_idx = None
        return gaps

    def publish_drive_command(self, steering_angle, speed):
        """
        Publish the AckermannDrive command to control the car.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(f"Published command -> Steering: {steering_angle:.2f}, Speed: {speed:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
