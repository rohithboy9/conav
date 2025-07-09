#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu

class low_pass_filter():
    def __init__(self):
        rospy.init_node("low_pass_filter")
        self.UAS1ImuMsg = None
        self.UAS1FirstImuMsg = True
        self.UAS1_imu_x = None
        self.UAS1_imu_y = None
        self.UAS1_imu_z = None
        self.UAS2ImuMsg = None
        self.UAS2FirstImuMsg = True
        self.UAS2_imu_x = None
        self.UAS2_imu_y = None
        self.UAS2_imu_z = None
        self.UAS3ImuMsg = None
        self.UAS3FirstImuMsg = True
        self.UAS3_imu_x = None
        self.UAS3_imu_y = None
        self.UAS3_imu_z = None
        self.omg = 0.99
        self.UAS1ImuSub = rospy.Subscriber('/UAS1/IMU', Imu, self.UAS1ImuCallback)
        self.UAS2ImuSub = rospy.Subscriber('/UAS2/IMU', Imu, self.UAS2ImuCallback)
        self.UAS3ImuSub = rospy.Subscriber('/UAS3/IMU', Imu, self.UAS3ImuCallback)
        self.UAS1ImuPub = rospy.Publisher('/UAS1_filtered_imu', Imu, queue_size=1)
        self.UAS2ImuPub = rospy.Publisher('/UAS2_filtered_imu', Imu, queue_size=1)
        self.UAS3ImuPub = rospy.Publisher('/UAS3_filtered_imu', Imu, queue_size=1)
        self.rate = rospy.Rate(50)

    def UAS1ImuCallback(self,msg):
        self.UAS1ImuMsg = msg

    def UAS2ImuCallback(self,msg):
        self.UAS2ImuMsg = msg

    def UAS3ImuCallback(self,msg):
        self.UAS3ImuMsg = msg

    def filter_measurement(self):
        while not rospy.is_shutdown():
            if self.UAS1FirstImuMsg == True:
                if self.UAS1ImuMsg is not None:
                    self.UAS1_imu_x = self.UAS1ImuMsg.linear_acceleration.x
                    self.UAS1_imu_y = self.UAS1ImuMsg.linear_acceleration.y
                    self.UAS1_imu_z = self.UAS1ImuMsg.linear_acceleration.z
                    self.UAS1FirstImuMsg = False
            else:
                if self.UAS1ImuMsg is not None:
                    self.UAS1_imu_x = self.omg*self.UAS1_imu_x + (1-self.omg)*self.UAS1ImuMsg.linear_acceleration.x
                    self.UAS1_imu_y = self.omg*self.UAS1_imu_y + (1-self.omg)*self.UAS1ImuMsg.linear_acceleration.y
                    self.UAS1_imu_z = self.omg*self.UAS1_imu_z + (1-self.omg)*self.UAS1ImuMsg.linear_acceleration.z

                    UAS1ImuPub = Imu()
                    UAS1ImuPub.header = self.UAS1ImuMsg.header
                    UAS1ImuPub.header.stamp = self.UAS1ImuMsg.header.stamp
                    UAS1ImuPub.linear_acceleration.x = self.UAS1_imu_x
                    UAS1ImuPub.linear_acceleration.y = self.UAS1_imu_y
                    UAS1ImuPub.linear_acceleration.z = self.UAS1_imu_z
                    UAS1ImuPub.angular_velocity.x = self.UAS1ImuMsg.angular_velocity.x
                    UAS1ImuPub.angular_velocity.y = self.UAS1ImuMsg.angular_velocity.y
                    UAS1ImuPub.angular_velocity.z = self.UAS1ImuMsg.angular_velocity.z
                    UAS1ImuPub.orientation.x = self.UAS1ImuMsg.orientation.x
                    UAS1ImuPub.orientation.y = self.UAS1ImuMsg.orientation.y
                    UAS1ImuPub.orientation.z = self.UAS1ImuMsg.orientation.z
                    UAS1ImuPub.orientation.w = self.UAS1ImuMsg.orientation.w
                    self.UAS1ImuPub.publish(UAS1ImuPub)

            if self.UAS2FirstImuMsg == True:
                if self.UAS2ImuMsg is not None:
                    self.UAS2_imu_x = self.UAS2ImuMsg.linear_acceleration.x
                    self.UAS2_imu_y = self.UAS2ImuMsg.linear_acceleration.y
                    self.UAS2_imu_z = self.UAS2ImuMsg.linear_acceleration.z
                    self.UAS2FirstImuMsg = False
            else:
                if self.UAS2ImuMsg is not None:
                    self.UAS2_imu_x = self.omg*self.UAS2_imu_x + (1-self.omg)*self.UAS2ImuMsg.linear_acceleration.x
                    self.UAS2_imu_y = self.omg*self.UAS2_imu_y + (1-self.omg)*self.UAS2ImuMsg.linear_acceleration.y
                    self.UAS2_imu_z = self.omg*self.UAS2_imu_z + (1-self.omg)*self.UAS2ImuMsg.linear_acceleration.z
                    
                    UAS2ImuPub = Imu()
                    UAS2ImuPub.header = self.UAS2ImuMsg.header
                    UAS2ImuPub.header.stamp = self.UAS2ImuMsg.header.stamp
                    UAS2ImuPub.linear_acceleration.x = self.UAS2_imu_x
                    UAS2ImuPub.linear_acceleration.y = self.UAS2_imu_y
                    UAS2ImuPub.linear_acceleration.z = self.UAS2_imu_z
                    UAS2ImuPub.angular_velocity.x = self.UAS2ImuMsg.angular_velocity.x
                    UAS2ImuPub.angular_velocity.y = self.UAS2ImuMsg.angular_velocity.y
                    UAS2ImuPub.angular_velocity.z = self.UAS2ImuMsg.angular_velocity.z
                    UAS2ImuPub.orientation.x = self.UAS2ImuMsg.orientation.x
                    UAS2ImuPub.orientation.y = self.UAS2ImuMsg.orientation.y
                    UAS2ImuPub.orientation.z = self.UAS2ImuMsg.orientation.z
                    UAS2ImuPub.orientation.w = self.UAS2ImuMsg.orientation.w
                    self.UAS2ImuPub.publish(UAS2ImuPub)

            if self.UAS3FirstImuMsg == True:
                if self.UAS3ImuMsg is not None:
                    self.UAS3_imu_x = self.UAS3ImuMsg.linear_acceleration.x
                    self.UAS3_imu_y = self.UAS3ImuMsg.linear_acceleration.y
                    self.UAS3_imu_z = self.UAS3ImuMsg.linear_acceleration.z
                    self.UAS3FirstImuMsg = False
            else:
                if self.UAS3ImuMsg is not None:
                    self.UAS3_imu_x = self.omg*self.UAS3_imu_x + (1-self.omg)*self.UAS3ImuMsg.linear_acceleration.x
                    self.UAS3_imu_y = self.omg*self.UAS3_imu_y + (1-self.omg)*self.UAS3ImuMsg.linear_acceleration.y
                    self.UAS3_imu_z = self.omg*self.UAS3_imu_z + (1-self.omg)*self.UAS3ImuMsg.linear_acceleration.z
                    
                    UAS3ImuPub = Imu()
                    UAS3ImuPub.header = self.UAS3ImuMsg.header
                    UAS3ImuPub.header.stamp = self.UAS3ImuMsg.header.stamp
                    UAS3ImuPub.linear_acceleration.x = self.UAS3_imu_x
                    UAS3ImuPub.linear_acceleration.y = self.UAS3_imu_y
                    UAS3ImuPub.linear_acceleration.z = self.UAS3_imu_z
                    UAS3ImuPub.angular_velocity.x = self.UAS3ImuMsg.angular_velocity.x
                    UAS3ImuPub.angular_velocity.y = self.UAS3ImuMsg.angular_velocity.y
                    UAS3ImuPub.angular_velocity.z = self.UAS3ImuMsg.angular_velocity.z
                    UAS3ImuPub.orientation.x = self.UAS3ImuMsg.orientation.x
                    UAS3ImuPub.orientation.y = self.UAS3ImuMsg.orientation.y
                    UAS3ImuPub.orientation.z = self.UAS3ImuMsg.orientation.z
                    UAS3ImuPub.orientation.w = self.UAS3ImuMsg.orientation.w
                    self.UAS3ImuPub.publish(UAS3ImuPub)

            self.rate.sleep()

def main():
    qe = low_pass_filter()
    qe.filter_measurement()

if __name__=="__main__":
    main()




