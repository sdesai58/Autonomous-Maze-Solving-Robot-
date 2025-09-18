#lab 4 lidar

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from std_msgs.msg import Int32
from rclpy.qos import *

class Lidar(Node):
    def __init__(self):
        super().__init__('robot_controller')
        lidar_qos = QoSProfile(depth = 1)
        lidar_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        lidar_qos.history = QoSHistoryPolicy.KEEP_LAST
        lidar_qos.durability = QoSDurabilityPolicy.VOLATILE

        self.laserSubscribe = self.create_subscription(LaserScan,'/scan',self.laser_callback,lidar_qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.DistancePublisher = self.create_publisher(Point,'/dist', 10)
    

    def laser_callback(self,scan):
        # scan = LaserScan   
        fullarrayR = np.arange(start=scan.angle_min,step=scan.angle_increment,stop=scan.angle_max)
        fullarrayD = fullarrayR*180/np.pi
        ranges = np.asarray(scan.ranges)
        mask1 = np.isnan(ranges)
        ranges[mask1] = 6
        # Front Lidar Data
        mask = (fullarrayD > 180)
        fullarrayD[mask] = fullarrayD[mask] - 360
        startang = -15
        endang = 15
        mask = (startang < fullarrayD ) * (fullarrayD < endang)
        DesarrayD = fullarrayD[mask]
        angle_increment = scan.angle_increment*180/np.pi
        objDist = []
        self.heading = 0.0
        for angle in DesarrayD:
            desranges = ranges[mask]
            index = int(np.round((angle - scan.angle_min)/angle_increment))
            index = np.clip(index, 0, len(desranges) - 1)
            objDist.append(desranges[index])
        objDist = (np.min(objDist))
        point = Point()
        if math.isnan(objDist):
             pass
        else:
            point.x = float(objDist)
        # self.get_logger().info('Front Object Distance: %s' % point.x)
        self.DistancePublisher.publish(point) 

        # # Side Lidar Data

        # mask = (fullarrayD > 180)
        # fullarrayD[mask] = fullarrayD[mask] - 360
        # startang = -135
        # endang = -45
        # mask = (startang < fullarrayD ) * (fullarrayD < endang)
        # DesarrayDSide = fullarrayD[mask]
    
        # objDistSide = []
        # self.heading = 0.0
        # for angle in DesarrayDSide:
        #     desrangesside = ranges[mask]
        #     index = int(np.round((angle - scan.angle_min)/angle_increment))
        #     index = np.clip(index, -(len(desrangesside) - 1), len(desrangesside) - 1)
        #     objDistSide.append(desrangesside[index])
        # objDistSide = (np.min(objDistSide))
        # if math.isnan(objDistSide):
        #      pass
        # else:
        #     point.y = float(objDistSide)
        # self.get_logger().info('Side Object Distance: %s' % point.y)
        # self.DistancePublisher.publish(point) 

def main():
	rclpy.init() #init routine needed for ROS2.
	objNode = Lidar() #Create class object to be used.
	rclpy.spin(objNode)
    

	#Clean up and shutdown.
	objNode.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
	main()
