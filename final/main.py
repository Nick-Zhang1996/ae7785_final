import rclpy
from rclpy.node import Node
from math import degrees,radians,cos,sin
import numpy as np
from time import sleep,time
from threading import Event,Thread
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

# when turning, make sure final position is aligned with nearby walls
# when moving, make sure to stop when facing a wall
# continuously run visualization algorithm, store the labels to increase confidence

class Main(Node):
    def __init__(self):
        super().__init__('set_goal')
        self.wall_dist_limit = 0.3


        # distance to wall in front
        # if no wall in sight this is set to 1.0
        self.wall_distance = 0
        self.angle_diff = 0
        self.camera_enable = True
        self.label_vec = []
        self.label_ts_vec = []
        self.label2text = ['empty','left','right','do not enter','stop','goal']




        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 1)
        self.listener = self.create_subscription(PointStamped, '/clicked_point',self.callback, 1)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 5)
        #Set up QoS Profiles for passing images over WiFi
        lidar_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1
        )


        self.sub_scan = self.create_subscription(LaserScan,'scan',self.lidar_callback,lidar_qos_profile)
        # for visualization
        self.fig = plt.gcf()
        self.ax = self.fig.gca()

    # lidar callback
    # sets: self.wall_distance -> distance to front wall
    # sets: self.angle_diff -> relevent angle from being aligned from the walls, in rad
    # TODO
    def lidar_callback(self,msg):
        angle_inc = (msg.angle_max-msg.angle_min) / len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        angles = (angles + np.pi) % (2*np.pi) - np.pi
        ranges = np.array(msg.ranges)
        return

    # camera callback
    # only runs when self.camera_enable = True
    # sets: self.label_vec, which is a FIFO queue storing identified labels
    # sets: self.label_ts_vec, which is time stamp for self.label_vec
    # TODO
    def camera_callback(self,msg):
        return

    # return the label the camera is looking at
    # only include results from past 0.5 seconds
    # only return if self.label_vec has a consistent reading
    # return None if can't make a decision
    # TODO
    def getLabel(self):
        return 0

    # take appropriate action given label
    def takeAction(self,label):
        #label2text = ['empty','left','right','do not enter','stop','goal']

        if (label is None):
            self.get_logger().info(f'Cannot find label, retrying ...')
            sleep(1)
            return

        self.get_logger().info(f'acting on label: {self.label2text[label]}')

        if (label == 0): # empty
            self.turnRight()
            self.align()
            self.goForward()
        elif (label == 1): # left
            self.turnLeft()
            self.align()
            self.goForward()
        elif (label == 2): # right
            self.turnRight()
            self.align()
            self.goForward()
        elif (label == 2 or label == 3): # do not enter / stop
            self.turnRight()
            self.align()
            self.turnRight()
            self.align()
            self.goForward()
        elif (label == 4): # goal
            pass

        return


    # start by driving forward
    # if a wall is encountered (closer than self.wall_dist_limit)
    # identify the sign, if it's goal -> terminate
    # turn accordingly (open loop)
    # find adjust turning to align with walls
    # repeat
    def run(self):
        self.goForward()
        label = self.getLabel()

        while (label != 4):
            self.takeAction(label)
        return

    # fine adjust orientation
    # TODO
    def align(self):
        return

    def turnLeft(self):
        dt_turn = 0.3
        self.get_logger().info(f'turning 90 deg ccw, open loop')
        msg = Twist()
        msg.angular.z = 0.3
        self.pub_cmd.publish(msg)
        sleep(radians(90)/0.3+dt_turn)
        msg = Twist()
        self.pub_cmd.publish(msg)
        sleep(0.1)
        return

    def turnRight(self):
        dt_turn = 0.3
        self.get_logger().info(f'turning 90 deg cw, open loop')
        msg = Twist()
        msg.angular.z = -0.3
        self.pub_cmd.publish(msg)
        sleep(radians(90)/0.3+dt_turn)
        msg = Twist()
        self.pub_cmd.publish(msg)
        sleep(0.1)
        return

    # go forward until close enough to a wall
    def goForward(self):
        v_linear = 0.15
        # go to first node
        self.get_logger().info(f'forwarding...')

        msg = Twist()
        msg.linear.x = v_linear
        self.publisher.publish(msg)

        while (self.wall_distance > self.wall_dist_limit):
            self.get_logger().info(f'wall dist {self.wall_distance}')
            sleep(0.2)

        msg = Twist()
        self.publisher.publish(msg)
        sleep(0.1)
        self.get_logger().info(f'stopped at {self.wall_distance}')
        return



def main(args=None):
    rclpy.init(args=args)
    node = Main()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
