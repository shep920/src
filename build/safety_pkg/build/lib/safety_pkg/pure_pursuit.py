#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Parameters
        self.declare_parameter('waypoints_file', 'waypoints.csv')
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('velocity', 0.2)
        self.declare_parameter('vehicle_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        # Get parameters
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.velocity = self.get_parameter('velocity').value
        self.vehicle_frame = self.get_parameter('vehicle_frame').value
        self.map_frame = self.get_parameter('map_frame').value

        # Load waypoints
        self.waypoints = self.load_waypoints(self.waypoints_file)
        if len(self.waypoints) == 0:
            self.get_logger().error('No waypoints loaded.')
            return

        # Publishers and Subscribers
        self.create_subscription(
            PoseStamped,
            '/current_pose',  # Replace with '/pf/pose/odom' for real car
            self.pose_callback,
            10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)

        self.current_pose = None
        self.current_waypoint_index = 0

        # Visualization
        self.publish_waypoints_markers()

        self.get_logger().info('Pure Pursuit node initialized.')

    def load_waypoints(self, filename):
        waypoints = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = line.split(',')
                        if len(values) >= 2:
                            x = float(values[0])
                            y = float(values[1])
                            waypoints.append([x, y])
            self.get_logger().info(f'{len(waypoints)} waypoints loaded.')
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
        return waypoints

    def pose_callback(self, msg):
        self.current_pose = msg.pose
        self.pure_pursuit_control()

    def pure_pursuit_control(self):
        if self.current_pose is None:
            return

        # Get current position
        position = self.current_pose.position
        x = position.x
        y = position.y

        # Find the target waypoint
        target_index = self.find_target_waypoint(x, y)
        if target_index is None:
            # Reached the end of waypoints
            self.get_logger().info('Reached the end of waypoints.')
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        target_waypoint = self.waypoints[target_index]

        # Compute control commands
        dx = target_waypoint[0] - x
        dy = target_waypoint[1] - y

        # Transform to vehicle coordinate frame
        yaw = self.get_yaw_from_pose(self.current_pose)
        transformed_point = self.transform_to_vehicle_frame(dx, dy, yaw)
        Ld = math.sqrt(transformed_point[0]**2 + transformed_point[1]**2)

        # Compute curvature γ = 2*y / Ld^2
        if Ld == 0:
            gamma = 0.0
        else:
            gamma = 2 * transformed_point[1] / (Ld**2)

        # Compute steering angle
        steer_angle = math.atan(gamma * self.lookahead_distance)

        # Publish command
        twist = Twist()
        twist.linear.x = self.velocity
        twist.angular.z = steer_angle
        self.cmd_pub.publish(twist)

    def find_target_waypoint(self, x, y):
        # Find the closest waypoint ahead of the vehicle at a lookahead distance
        min_distance = float('inf')
        target_index = None
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            waypoint = self.waypoints[i]
            distance = math.hypot(waypoint[0] - x, waypoint[1] - y)
            if distance >= self.lookahead_distance and distance < min_distance:
                min_distance = distance
                target_index = i

        if target_index is not None:
            self.current_waypoint_index = target_index
        return target_index

    def get_yaw_from_pose(self, pose):
        # Convert quaternion to yaw
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def transform_to_vehicle_frame(self, dx, dy, yaw):
        # Transform point to vehicle frame
        x_v = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_v = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return [x_v, y_v]

    def publish_waypoints_markers(self):
        marker_array = MarkerArray()
        for idx, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'waypoints'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuitNode()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
