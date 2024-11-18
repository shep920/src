#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import csv
import math
from visualization_msgs.msg import Marker

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        # Parameters
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('waypoints_file', 'waypoints.csv')
        self.declare_parameter('max_steering_angle', 0.34)
        self.declare_parameter('velocity', 1.0)

        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.velocity = self.get_parameter('velocity').value

        # Load waypoints
        self.waypoints = self.load_waypoints(self.waypoints_file)

        # Subscribers
        self.pose_sub = self.create_subscription(
            Odometry,  # Use Odometry for simulation or PoseStamped for particle filter
            '/odom',  # '/pf/viz/inferred_pose' for particle filter
            self.pose_callback,
            10
        )

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # Visualization Publisher
        self.marker_pub = self.create_publisher(
            Marker,
            '/lookahead_point',
            10
        )

        self.current_pose = None

    def load_waypoints(self, filename):
        waypoints = []
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    x, y = float(row[0]), float(row[1])
                    waypoints.append((x, y))
            self.get_logger().info(f"Loaded {len(waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
        return waypoints

    def pose_callback(self, msg):
        # Get current position
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert orientation to yaw angle
        yaw = self.quaternion_to_euler(orientation)

        self.current_pose = (position.x, position.y, yaw)

        # Proceed if waypoints are loaded
        if self.waypoints:
            self.pure_pursuit_control()

    def pure_pursuit_control(self):
        # Find the lookahead point
        lookahead_point = self.get_lookahead_point(self.current_pose, self.waypoints, self.lookahead_distance)
        if lookahead_point is None:
            self.get_logger().info("No valid lookahead point found.")
            return

        # Transform lookahead point to vehicle coordinates
        ld_x, ld_y = self.transform_point(self.current_pose, lookahead_point)

        # Compute steering angle
        steering_angle = math.atan2(2 * ld_y * self.velocity, self.lookahead_distance ** 2)
        steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)

        # Publish drive command
        self.publish_drive_command(steering_angle, self.velocity)

        # Visualize lookahead point
        self.publish_lookahead_marker(lookahead_point)

    def get_lookahead_point(self, pose, waypoints, lookahead_distance):
        position = np.array([pose[0], pose[1]])
        for i in range(len(waypoints)):
            wp = np.array(waypoints[i])
            dist = np.linalg.norm(wp - position)
            if dist >= lookahead_distance:
                return waypoints[i]
        return waypoints[-1]  # Return the last waypoint if none is further than lookahead_distance

    def transform_point(self, pose, point):
        x, y, yaw = pose
        dx = point[0] - x
        dy = point[1] - y

        # Rotate the point into the vehicle's coordinate frame
        transformed_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        transformed_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        return transformed_x, transformed_y

    def quaternion_to_euler(self, orientation):
        # Convert quaternion to euler yaw angle
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def publish_drive_command(self, steering_angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(f"Steering Angle: {steering_angle:.2f}, Speed: {speed:.2f}")

    def publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
