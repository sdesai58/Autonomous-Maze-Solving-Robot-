import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Int8
import time
from scipy import stats

# etc
import numpy as np
from numpy.linalg import norm

import math


class print_transformed_odom(Node):
    def __init__(self):
        super().__init__('print_fixed_odom')
        # State (for the update_Odometry code)
        #self.waypoints = np.array([1.5, 0.0], [1.5, 1.4], [0, 1.4])
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.twistPub = Twist()
        self.case = 6
        self.turncase = 0
        self.desired_target_generated = False
        self.current_target = 0
        self.ret = 0
        self.counter = 0
        self.lastret = 0
        # self.desired_target = None
        # self.desired_target2 = None

        self.odom_sub = self.create_subscription(Odometry,'/odom',self.odom_callback,1)
        self.vel_publish= self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Point,'/dist',self.lidar_callback,10)
        self.ret_sub = self.create_subscription(Int8,'/ret', self.ret_callback,10)
        self.odom_sub  # prevent unused variable warning
        self.lidardist = Point()


    def lidar_callback(self, stupid):
        self.lidardist = stupid

    def odom_callback(self, data):
        self.update_Odometry(data)

    def update_Odometry(self,Odom):
        self.get_logger().info('Case: %s' % self.case)
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
    
        # self.get_logger().info('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
        # self.get_logger().info('Lidar Distance: %s' % self.lidardist.x)
        # self.get_logger().info('Lidar Distance: %s' % self.case)
    
        #Case 1. We are moving from start to point 1

        if self.case is 0:
            if not self.desired_target_generated:
                self.current_target = self.globalAng - np.pi / 100
                self.get_logger().info('Current Target: %s' % self.current_target)
                if self.current_target > np.pi:
                    self.current_target -= 2 * np.pi
                elif self.current_target < -np.pi:
                    self.current_target += 2 * np.pi
                self.desired_target_generated = True
            angle_error = self.current_target - self.globalAng
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            angle_tolerance = np.deg2rad(3)
            k = 0.5
            if abs(angle_error) > angle_tolerance:
                self.twistPub.angular.z = k * angle_error
                self.vel_publish.publish(self.twistPub)
            else:
                self.twistPub.angular.z = 0.0
                self.vel_publish.publish(self.twistPub)
                self.case = 6
                self.desired_target_generated = False



        if self.case == 1:
            if not self.desired_target_generated:
                self.current_target = self.globalAng + np.pi / 2
                self.get_logger().info('Current Target: %s' % self.current_target)
                if self.current_target > np.pi:
                    self.current_target -= 2 * np.pi
                elif self.current_target < -np.pi:
                    self.current_target += 2 * np.pi
                self.desired_target_generated = True
            self.get_logger().info('Im in case 1!')
            angle_error = self.current_target - self.globalAng
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            angle_tolerance = np.deg2rad(3)
            k = 0.5
            if abs(angle_error) > angle_tolerance:
                self.twistPub.angular.z = k * angle_error
                self.vel_publish.publish(self.twistPub)
            else:
                self.twistPub.angular.z = 0.0
                self.vel_publish.publish(self.twistPub)
                self.case = 6
                self.desired_target_generated = False

        if self.case == 2:
            if not self.desired_target_generated:
                self.current_target = self.globalAng - np.pi / 2
                self.get_logger().info('Current Target: %s' % self.current_target)
                if self.current_target > np.pi:
                    self.current_target -= 2 * np.pi
                elif self.current_target < -np.pi:
                    self.current_target += 2 * np.pi
                self.desired_target_generated = True
            angle_error = self.current_target - self.globalAng
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            angle_tolerance = np.deg2rad(3)
            k = 0.5
            if abs(angle_error) > angle_tolerance:
                self.twistPub.angular.z = k * angle_error
                self.vel_publish.publish(self.twistPub)
            else:
                self.twistPub.angular.z = 0.0
                self.vel_publish.publish(self.twistPub)
                self.case = 6
                self.desired_target_generated = False

        if self.case == 3 or self.case == 4:
            if not self.desired_target_generated:
                self.current_target = self.globalAng + np.pi
                self.get_logger().info('Current Target: %s' % self.current_target)
                if self.current_target > np.pi:
                    self.current_target -= 2 * np.pi
                elif self.current_target < -np.pi:
                    self.current_target += 2 * np.pi
                self.desired_target_generated = True
            angle_error = self.current_target - self.globalAng
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            angle_tolerance = np.deg2rad(3)
            k = 0.5
            if abs(angle_error) > angle_tolerance:
                self.twistPub.angular.z = k * angle_error
                self.vel_publish.publish(self.twistPub)
            else:
                self.twistPub.angular.z = 0.0
                self.vel_publish.publish(self.twistPub)
                self.case = 6
                self.desired_target_generated = False

        if self.case == 5:
            self.twistPub.linear.x = 0.0
            self.twistPub.angular.z = 0.0
            self.vel_publish.publish(self.twistPub)
            self.get_logger().info('Burger is happy! Burger made it to star')

        if self.case == 6:
            self.twistPub.linear.x = 0.1
            self.vel_publish.publish(self.twistPub)                
            if(self.lidardist.x) < 0.55 and (self.lidardist.x) > 0.0:
                self.twistPub.linear.x = 0.0
                self.vel_publish.publish(self.twistPub)
                self.case = 7
                self.desired_target_generated = False

        # if self.case == :
        #     temp = np.zeros(10)
        #     for i in range(10):
        #        self.get_logger().info('Current Ret: %s' % self.ret)
        #        temp[i] = int(self.ret)
        #     self.ret = stats.mode(temp)
        #     self.case = 8

        if self.case == 7:
            if self.ret == self.lastret:
                self.counter += 1
                self.get_logger().info('Counter: %s' % self.counter)
            else:
                self.counter = 0
            self.lastret = self.ret
            # self.get_logger().info('Current Ret: %s' % self.ret)
            if self.counter == 3:
                self.counter = 0
                if self.ret == 0:
                    self.case = 0
                elif self.ret == 1:
                    self.case = 1
                elif self.ret == 2:
                    self.case = 2 
                elif self.ret == 3:
                    self.case = 3
                elif self.ret == 4:
                    self.case = 4
                elif self.ret == 5:
                    self.case = 5



    def ret_callback(self,msg):
        self.ret = msg.data
        
        
def main(args=None):
    rclpy.init(args=args)
    print_odom = print_transformed_odom()
    rclpy.spin(print_odom)
    print_odom.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()





    
