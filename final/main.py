import rclpy
from rclpy.node import Node
from math import degrees,radians,cos,sin
import numpy as np
from time import sleep,time
from threading import Event,Thread
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Point,Twist,Pose,PoseWithCovariance
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from identify import Signs
import cv2

# when turning, make sure final position is aligned with nearby walls
# when moving, make sure to stop when facing a wall
# continuously run visualization algorithm, store the labels to increase confidence

class Main(Node):
    def __init__(self):
        super().__init__('main')
        self.wall_dist_limit = 0.5
        self.lidar_angle_deg = 4

        self.br = CvBridge()
        self.vision = Signs()
        self.vision.loadModel()
        self.start_labeling = Event()

        # distance to wall in front
        # if no wall in sight this is set to 1.0
        self.wall_distance = 0
        self.angle_diff = 0
        self.camera_enable = True
        self.label_vec = []
        self.label_ts_vec = []
        self.label2text = ['empty','left','right','do not enter','stop','goal']

        # temporary
        self.label = 0

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 5)
        #Set up QoS Profiles for passing images over WiFi
        lidar_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1
        )


        self.sub_scan = self.create_subscription(LaserScan,'scan',self.lidar_callback,lidar_qos_profile)
        self.sub_camera = self.create_subscription(CompressedImage,'/camera/image/compressed',self.camera_callback,lidar_qos_profile)
        # for visualization
        self.fig = plt.gcf()
        self.ax = self.fig.gca()

    def show_scan(self, angles, ranges, mask):
        ax = self.ax
        ax.cla()
        scan_x = np.cos(angles) * ranges
        scan_y = np.sin(angles) * ranges
        ax.scatter(scan_x,scan_y,color='k')

        # ROI
        scan_x = np.cos(angles[mask]) * ranges[mask]
        scan_y = np.sin(angles[mask]) * ranges[mask]
        ax.scatter(scan_x,scan_y,color='b')

        # robot location
        circle = plt.Circle((0,0),0.1, color='r')
        ax.add_patch(circle)
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal','box')

        # robot FOV (ROI)
        a = self.lidar_angle_deg
        ax.plot([0, cos(radians(a))],[0, sin(radians(a))] )
        ax.plot([0, cos(radians(-a))],[0, sin(-radians(a))] )

        return

    # lidar callback
    # sets: self.wall_distance -> distance to front wall
    # sets: self.angle_diff -> relevent angle from being aligned from the walls, in rad
    # TODO
    def lidar_callback(self,msg):
        angle_inc = (msg.angle_max-msg.angle_min) / len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        angles = (angles + np.pi) % (2*np.pi) - np.pi
        ranges = np.array(msg.ranges)

        # set self.wall_distance, self.angle_diff
        self.process_lidar(angles, ranges)
        return

    # conduct hough transform to find all lines nearby
    # then return wall distance and angle misalignment
    def process_lidar(self, angles, ranges):
        # Task 1 ---- find wall distance
        # first focus on straight ahead +/-  10 degs

        a = self.lidar_angle_deg
        mask = np.bitwise_and(angles < radians(a), angles > radians(-a))
        mask = np.bitwise_and(mask, np.invert(np.isnan(ranges)))
        xx = ranges[mask]*np.cos(angles[mask])
        yy = ranges[mask]*np.sin(angles[mask])

        #self.show_scan(angles, ranges, mask)
        
        # TODO check dimension
        points = np.vstack([xx,yy]).T
        # d = x cos + y sin, gives theta, d
        line = self.hough(points,n=1)[0]
        # is the wall mostly perpendicular?
        # theta=0 -> directly in front, theta>0, leaning to second quadrant
        # how far is the wall
        if (np.abs(line[0]) < radians(10)):
            self.wall_distance = line[1]
        else:
            self.wall_distance = 1.0
        #print(f'theta = {degrees(line[0])}, d = {line[1]}')

        # plot the said line
        theta = (line[0] + np.pi/2) % np.pi - np.pi/2
        d = line[1]
        #xx = np.linspace(-1,1)
        #yy = (d-np.cos(theta)*xx)/np.sin(theta)
        #self.ax.plot(xx,yy)

        # Task 2 ---- find misalignment
        mask = ranges < 2
        xx = ranges[mask]*np.cos(angles[mask])
        yy = ranges[mask]*np.sin(angles[mask])
        points = np.vstack([xx,yy]).T
        # d = x cos + y sin, gives theta, d
        lines = self.hough(points,n=1)
        thetas = np.array([line[0] for line in lines])
        # wrap to -np.pi, np.pi
        thetas = (thetas + np.pi/4) % (np.pi/2) - np.pi/4
        angle_disagreement = np.max(thetas) - np.min(thetas)
        self.angle_diff = np.mean(thetas)
        wrapped_wall_angle = (line[0] + np.pi/2) % np.pi - np.pi/2
        #self.get_logger().info(f'wall: {degrees(wrapped_wall_angle):.2f}deg, d={line[1]}, misalign = {degrees(self.angle_diff):.2f}deg, (err {angle_disagreement})')
        # doesn't work in multithreading
        #plt.pause(0.05)

        #fig.canvas.draw()
        #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #cv2.imshow('debug',data)

        
        return
        

    # conduct hough transform, give equations for the top n results
    # d = cos(theta) * x + sin(theta) * y
    # line parameter: (theta, d)
    def hough(self, points, n=3):
        # d: [0-0.01m, 0.01-0.02m, ... - 1m]
        # theta [0-1deg, 1-2deg, ... 180deg]
        # panel[theta_idx, d_idx]
        dd = 0.01
        dtheta = radians(1)
        d_count = int(1.0/dd)+1
        theta_count = int(np.pi/dtheta)+1
        panel = np.zeros((theta_count,d_count),dtype=int)
        # TODO verify spacing and slots
        thetas = np.linspace(-np.pi/2,np.pi/2, theta_count)
        ds = np.linspace(0,1, d_count)
        for point in points:
            d_vec = point[0] * np.cos(thetas) + point[1] * np.sin(thetas)
            for i in range(theta_count):
                try:
                    panel[i,int(d_vec[i]/dd)] += 1
                except IndexError:
                    pass
        #indices = np.unravel_index(np.argmax(panel,axis=None), panel.shape)
        #return ( thetas[indices[0]], ds[indices[1] ])

        #2*n incices of 3 best candidates
        multiple_indices = np.unravel_index(np.argpartition(panel.flatten(),-n)[-n:], panel.shape)
        
        theta_d = [( thetas[multiple_indices[0][i]], ds[multiple_indices[1][i] ]) for i in range(len(multiple_indices[0]))]
        return theta_d
        
    # camera callback
    # only runs when self.camera_enable = True
    # sets: self.label_vec, which is a FIFO queue storing identified labels
    # sets: self.label_ts_vec, which is time stamp for self.label_vec
    def camera_callback(self,msg):
        img = self.br.compressed_imgmsg_to_cv2(msg)
        if (img is None):
            self.get_logger().info(f'[camera_callback] got None')
        label = self.vision.predict(img)
        self.label = label
        if (self.start_labeling.is_set()):
            self.label_vec.append(label)

        self.get_logger().info(f'[camera_callback] found label: {self.label2text[label]}')

        #cv2.imshow('debug',img)
        #key = cv2.waitKey(10)
        #if (key == 27):
        #    exit(0)

        return

    # return the label the camera is looking at
    # only include results from past 0.5 seconds
    # only return if self.label_vec has a consistent reading
    # return None if can't make a decision
    def getLabel(self):
        self.get_logger().info(f'identifying labels, 2 sec...')

        self.label_vec = []
        self.start_labeling.set()
        sleep(2)
        self.start_labeling.clear()
        values, counts = np.unique(self.label_vec, return_counts=True)
        idx = np.argmax(counts)
        label = values[idx]
        confidence = counts[idx]/np.sum(counts)

        self.get_logger().info(f'label = {self.label2text[label]}, confidence = {confidence}')

        return label

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
            return

        return


    # start by driving forward
    # if a wall is encountered (closer than self.wall_dist_limit)
    # identify the sign, if it's goal -> terminate
    # turn accordingly (open loop)
    # find adjust turning to align with walls
    # repeat
    def run(self):
        try:
            self.goForward()
            label = self.getLabel()

            while (label != 4):
                self.takeAction(label)
                label = self.getLabel()
        except KeyboardInterrupt:
            msg = Twist()
            self.pub_cmd.publish(msg)
            sleep(0.1)
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
        self.get_logger().info(f'done')
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
        self.get_logger().info(f'done')
        return

    # go forward until close enough to a wall
    def goForward(self):
        v_linear = 0.15
        # go to first node
        self.get_logger().info(f'forwarding...')

        msg = Twist()
        msg.linear.x = v_linear
        self.pub_cmd.publish(msg)

        while (self.wall_distance > self.wall_dist_limit):
            #self.get_logger().info(f'wall dist {self.wall_distance}')
            msg = Twist()
            msg.linear.x = v_linear
            self.pub_cmd.publish(msg)
            sleep(0.2)

        msg = Twist()
        self.pub_cmd.publish(msg)
        sleep(0.1)
        self.get_logger().info(f'stopped at {self.wall_distance}')
        return



def main(args=None):
    rclpy.init(args=args)
    node = Main()

    thread = Thread(target=rclpy.spin, args=(node,),daemon=True)
    thread.start()
    node.run()
    #rclpy.spin(node)

    thread.join()
    node.destroy_node()
    rclpy.shutdown()

def debug(args=None):
    rclpy.init(args=args)
    node = Main()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

def debug2(args=None):
    rclpy.init(args=args)
    main = Main()

    theta = radians(15)
    d = 0.4
    print(f'theta = {theta}, d = {d}')
    xx = np.linspace(3,5)
    yy = (d-np.cos(theta)*xx)/np.sin(theta)
    points1 = np.vstack([xx,yy]).T

    theta = radians(40)
    d = 0.8
    print(f'theta = {theta}, d = {d}')
    xx = np.linspace(-1,1)
    yy = (d-np.cos(theta)*xx)/np.sin(theta)
    points2 = np.vstack([xx,yy]).T

    theta = radians(2)
    d = 0.1
    print(f'theta = {theta}, d = {d}')
    xx = np.linspace(-1,1)
    yy = (d-np.cos(theta)*xx)/np.sin(theta)
    points3 = np.vstack([xx,yy]).T

    points = np.vstack([points1, points2, points3])

    val = main.hough(points,n=2)
    print(val)



if __name__ == '__main__':
    main()
    #debug()
