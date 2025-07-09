#!/usr/bin/env python3

# Cooperative Localization for three hexacopters "UAS1", "UAS2" and "UAS3"
# Estimated States: pn pe pd psi u v w acc_b1 acc_b2 acc_b3 rho_b1 rho_b2 rho_b3 rho_b4
# External measurements "GPS", "Velocity", "Range", "Range-rate"
# Internal measurements ax, ay, az, phi, theta, p, q, r

import rospy
import numpy as np
from numpy import sin, cos, tan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Range
from mavros_msgs.msg import GPSRAW
from sympy import Matrix
from localization.msg import localization_estimates
from localization.msg import localization_error
import utm
from dataclasses import dataclass

class StateEstimator():
    def __init__(self):
        rospy.init_node("Seven_State_Estimator")
        rospy.loginfo("Cooperative Localization node initialized")

        # Intialize Variables
        self.UAS1_FirstState     = True
        self.UAS2_FirstState     = True
        self.UAS3_FirstState     = True
        self.UAS1_FirstGPS       = True
        self.UAS2_FirstGPS       = True
        self.UAS3_FirstGPS       = True
        self.NumStates           = 7+3+5+2
        self.NumVeh              = 3
        ns                       = self.NumStates
        self.xHat                = np.zeros((3*ns,1), dtype=np.float64)
        self.pCov                = 5.0*np.identity(3*ns)
        self.pCov[ns*0+3,ns*0+3] = 0.1
        self.pCov[ns*1+3,ns*1+3] = 0.1
        self.pCov[ns*2+3,ns*2+3] = 0.1
        self.local_origin        = None
        self.UAS1_Time           = None
        self.UAS2_Time           = None
        self.UAS3_Time           = None

        # Measurement Flags
        self.UAS1_GPS_flag        = False
        self.UAS2_GPS_flag        = False
        self.UAS3_GPS_flag        = False
        self.heightFlag           = True
        self.UAS1_LM1_flag        = True
        self.UAS2_LM1_flag        = True
        self.UAS3_LM1_flag        = True
        self.UAS1_LM2_flag        = True
        self.UAS2_LM2_flag        = True
        self.UAS3_LM2_flag        = True
        self.UAS1_LM3_flag        = True
        self.UAS2_LM3_flag        = True
        self.UAS3_LM3_flag        = True
        self.UAS1_LM4_flag        = True
        self.UAS2_LM4_flag        = True
        self.UAS3_LM4_flag        = True
        self.UAS1_LM5_flag        = True
        self.UAS2_LM5_flag        = True
        self.UAS3_LM5_flag        = True
        self.UAS1_LM6_flag        = False
        self.UAS2_LM6_flag        = False
        self.UAS3_LM6_flag        = False
        self.UAS1_2_UAS2_Flag     = True
        self.UAS2_2_UAS3_Flag     = True
        self.UAS1_2_UAS3_Flag     = True
        self.odomFlag             = False

        self.rhoMax               = 75.0 * 1000.0

        # Noise Parameters
        self.axSigma              = 10.0
        self.aySigma              = 10.0
        self.azSigma              = 10.0
        self.phiSigma             = 0.2
        self.thetaSigma           = 0.2
        self.omgSigma             = 0.1
        self.gpsSigma             = 0.5
        self.rhoSigma             = 5
        self.rhoTimeDiff          = 0.1
        self.velSigma             = 0.5
        self.rhodotSigma          = 8

        # Subscriber messages
        self.UAS1_ImuMsg          = None
        self.UAS1_OdomMsg         = None
        self.UAS1_GPSMsg          = None
        self.UAS2_ImuMsg          = None
        self.UAS2_OdomMsg         = None
        self.UAS2_GPSMsg          = None
        self.UAS3_ImuMsg          = None
        self.UAS3_OdomMsg         = None
        self.UAS3_GPSMsg          = None

        self.lm1_to_UAS1_msg      = None
        self.lm2_to_UAS1_msg      = None
        self.lm3_to_UAS1_msg      = None
        self.lm4_to_UAS1_msg      = None
        self.lm5_to_UAS1_msg      = None
        self.lm6_to_UAS1_msg      = None
        self.lm1_to_UAS2_msg      = None
        self.lm2_to_UAS2_msg      = None
        self.lm3_to_UAS2_msg      = None
        self.lm4_to_UAS2_msg      = None
        self.lm5_to_UAS2_msg      = None
        self.lm6_to_UAS2_msg      = None
        self.lm1_to_UAS3_msg      = None
        self.lm2_to_UAS3_msg      = None
        self.lm3_to_UAS3_msg      = None
        self.lm4_to_UAS3_msg      = None
        self.lm5_to_UAS3_msg      = None
        self.lm6_to_UAS3_msg      = None
        self.UAS3_to_UAS1_msg     = None
        self.UAS2_to_UAS1_msg     = None
        self.UAS1_to_UAS2_msg     = None
        self.UAS3_to_UAS2_msg     = None
        self.UAS1_to_UAS3_msg     = None
        self.UAS2_to_UAS3_msg     = None


        # Subscribers
        self.UAS1_ImuSub          = rospy.Subscriber('/UAS1_filtered_imu', Imu, self.UAS1_ImuCallback)
        self.UAS1_OdomSub         = rospy.Subscriber('/UAS1/odom', Odometry, self.UAS1_OdomCallback)
        self.UAS1_GPSSub          = rospy.Subscriber('/UAS1/GPS', NavSatFix, self.UAS1_GPSCallback)
        self.UAS2_ImuSub          = rospy.Subscriber('/UAS2_filtered_imu', Imu, self.UAS2_ImuCallback)
        self.UAS2_OdomSub         = rospy.Subscriber('/UAS2/odom', Odometry, self.UAS2_OdomCallback)
        self.UAS2_GPSSub          = rospy.Subscriber('/UAS2/GPS', NavSatFix, self.UAS2_GPSCallback)
        self.UAS3_ImuSub          = rospy.Subscriber('/UAS3_filtered_imu', Imu, self.UAS3_ImuCallback)
        self.UAS3_OdomSub         = rospy.Subscriber('/UAS3/odom', Odometry, self.UAS3_OdomCallback)
        self.UAS3_GPSSub          = rospy.Subscriber('/UAS3/GPS', NavSatFix, self.UAS3_GPSCallback)

        self.lm1_to_UAS1_sub      = rospy.Subscriber('/range_lm1_to_UAS1', Range, self.lm1_to_UAS1_callback)
        self.lm2_to_UAS1_sub      = rospy.Subscriber('/range_lm2_to_UAS1', Range, self.lm2_to_UAS1_callback)
        self.lm3_to_UAS1_sub      = rospy.Subscriber('/range_lm3_to_UAS1', Range, self.lm3_to_UAS1_callback)
        self.lm4_to_UAS1_sub      = rospy.Subscriber('/range_lm4_to_UAS1', Range, self.lm4_to_UAS1_callback)
        self.lm5_to_UAS1_sub      = rospy.Subscriber('/range_lm5_to_UAS1', Range, self.lm5_to_UAS1_callback)
        self.lm6_to_UAS1_sub      = rospy.Subscriber('/range_lm6_to_UAS1', Range, self.lm6_to_UAS1_callback)
        self.lm1_to_UAS2_sub      = rospy.Subscriber('/range_lm1_to_UAS2', Range, self.lm1_to_UAS2_callback)
        self.lm2_to_UAS2_sub      = rospy.Subscriber('/range_lm2_to_UAS2', Range, self.lm2_to_UAS2_callback)
        self.lm3_to_UAS2_sub      = rospy.Subscriber('/range_lm3_to_UAS2', Range, self.lm3_to_UAS2_callback)
        self.lm4_to_UAS2_sub      = rospy.Subscriber('/range_lm4_to_UAS2', Range, self.lm4_to_UAS2_callback)
        self.lm5_to_UAS2_sub      = rospy.Subscriber('/range_lm5_to_UAS2', Range, self.lm5_to_UAS2_callback)
        self.lm6_to_UAS2_sub      = rospy.Subscriber('/range_lm6_to_UAS2', Range, self.lm6_to_UAS2_callback)
        self.lm1_to_UAS3_sub      = rospy.Subscriber('/range_lm1_to_UAS3', Range, self.lm1_to_UAS3_callback)
        self.lm2_to_UAS3_sub      = rospy.Subscriber('/range_lm2_to_UAS3', Range, self.lm2_to_UAS3_callback)
        self.lm3_to_UAS3_sub      = rospy.Subscriber('/range_lm3_to_UAS3', Range, self.lm3_to_UAS3_callback)
        self.lm4_to_UAS3_sub      = rospy.Subscriber('/range_lm4_to_UAS3', Range, self.lm4_to_UAS3_callback)
        self.lm5_to_UAS3_sub      = rospy.Subscriber('/range_lm5_to_UAS3', Range, self.lm5_to_UAS3_callback)
        self.lm6_to_UAS3_sub      = rospy.Subscriber('/range_lm6_to_UAS3', Range, self.lm6_to_UAS3_callback)

        self.UAS2_to_UAS1_sub     = rospy.Subscriber('/range_UAS2_to_UAS1', Range, self.UAS2_to_UAS1_callback)
        self.UAS3_to_UAS1_sub     = rospy.Subscriber('/range_UAS3_to_UAS1', Range, self.UAS3_to_UAS1_callback)
        self.UAS1_to_UAS2_sub     = rospy.Subscriber('/range_UAS1_to_UAS2', Range, self.UAS1_to_UAS2_callback)
        self.UAS3_to_UAS2_sub     = rospy.Subscriber('/range_UAS3_to_UAS2', Range, self.UAS3_to_UAS2_callback)
        self.UAS1_to_UAS2_sub     = rospy.Subscriber('/range_UAS1_to_UAS3', Range, self.UAS1_to_UAS3_callback)
        self.UAS3_to_UAS2_sub     = rospy.Subscriber('/range_UAS2_to_UAS3', Range, self.UAS2_to_UAS3_callback)

        # Publishers
        self.UAS1_EstPub          = rospy.Publisher('/UAS1_estimated_states', localization_estimates, queue_size=1)
        self.UAS1_ErrPub          = rospy.Publisher('/UAS1_error_states', localization_error, queue_size=1)
        self.UAS1_GPSPub          = rospy.Publisher('/UAS1_true_states', localization_estimates, queue_size=1)
        self.UAS2_EstPub          = rospy.Publisher('/UAS2_estimated_states', localization_estimates, queue_size=1)
        self.UAS2_ErrPub          = rospy.Publisher('/UAS2_error_states', localization_error, queue_size=1)
        self.UAS2_GPSPub          = rospy.Publisher('/UAS2_true_states', localization_estimates, queue_size=1)
        self.UAS3_EstPub          = rospy.Publisher('/UAS3_estimated_states', localization_estimates, queue_size=1)
        self.UAS3_ErrPub          = rospy.Publisher('/UAS3_error_states', localization_error, queue_size=1)
        self.UAS3_GPSPub          = rospy.Publisher('/UAS3_true_states', localization_estimates, queue_size=1)

        # rate
        self.rate         = rospy.Rate(20)
        self.steps        = 10
        self.gravity      = 9.81

    def setLocalOrigin(self):
        # Update these home location coordinates from test site
        org_lat           = 39.1549318997368
        org_lon           = -84.7890753969889
        org_alt           = 131.987901926041
        org_pos           = utm.from_latlon(org_lat,org_lon)
        self.local_origin = Local_Coordinate(org_pos[0], org_pos[1], org_alt, org_pos[2], org_pos[3])
        rospy.loginfo("Local Origin is set")

    def setLandmarkPosition(self):
        # Update these from test site
        lm1_lat  = 39.1549538
        lm1_lon  = -84.7883206
        lm1_alt  = 166.080 + 1.74
        lm1_pos  = utm.from_latlon(lm1_lat,lm1_lon)
        lm1_x, lm1_y, lm1_z = self.convert_to_local(lm1_pos[0], lm1_pos[1], lm1_alt)
        self.lm1 = np.array([lm1_x, lm1_y, lm1_z], dtype=np.float64)
        rospy.loginfo("Landmark 1 position is set")
        print("Landmark 1:",self.lm1)

        lm2_lat = 39.1547440
        lm2_lon = -84.7880159
        lm2_alt = 166.700 + 1.605
        lm2_pos  = utm.from_latlon(lm2_lat,lm2_lon)
        lm2_x, lm2_y, lm2_z = self.convert_to_local(lm2_pos[0], lm2_pos[1], lm2_alt)
        self.lm2 = np.array([lm2_x, lm2_y, lm2_z], dtype=np.float64)
        rospy.loginfo("Landmark 2 position is set")
        print("Landmark 2:",self.lm2)

        lm3_lat = 39.1546233
        lm3_lon = -84.7885084
        lm3_alt = 165.020 + 1.79
        lm3_pos  = utm.from_latlon(lm3_lat,lm3_lon)
        lm3_x, lm3_y, lm3_z = self.convert_to_local(lm3_pos[0], lm3_pos[1], lm3_alt)
        self.lm3 = np.array([lm3_x, lm3_y, lm3_z], dtype=np.float64)
        rospy.loginfo("Landmark 3 position is set")
        print("Landmark 3:",self.lm3)

        lm4_lat = 39.1544874
        lm4_lon = -84.7888732
        lm4_alt = 164.540 + 1.75
        lm4_pos  = utm.from_latlon(lm4_lat,lm4_lon)
        lm4_x, lm4_y, lm4_z = self.convert_to_local(lm4_pos[0], lm4_pos[1], lm4_alt)
        self.lm4 = np.array([lm4_x, lm4_y, lm4_z], dtype=np.float64)
        rospy.loginfo("Landmark 4 position is set")
        print("Landmark 4:",self.lm4)

        lm5_lat = 39.1542826
        lm5_lon = -84.7885955
        lm5_alt = 165.230 + 1.62
        lm5_pos  = utm.from_latlon(lm5_lat,lm5_lon)
        lm5_x, lm5_y, lm5_z = self.convert_to_local(lm5_pos[0], lm5_pos[1], lm5_alt)
        self.lm5 = np.array([lm5_x, lm5_y, lm5_z], dtype=np.float64)
        rospy.loginfo("Landmark 5 position is set")
        print("Landmark 5:",self.lm5)

    def convert_to_local(self, pos_x, pos_y, pos_z):
        return pos_x-self.local_origin.x, pos_y-self.local_origin.y, pos_z-self.local_origin.z

    def convert_to_global(self, pos_x, pos_y, pos_z):
        pos_x = pos_x + self.local_origin.x
        pos_y = pos_y + self.local_origin.y
        pos_z = pos_z + self.local_origin.z

        lat, lon = utm.to_latlon(pos_x, pos_y, self.local_origin.zone, self.local_origin.letter)
        return lat, lon, pos_z

    def setFirstGPS(self,msg):
        if msg is not None:
            lat = msg.latitude
            lon = msg.longitude
            alt = msg.altitude
            pos = utm.from_latlon(lat,lon)
            xPos, yPos, zPos = self.convert_to_local(pos[0],pos[1],alt)
            return np.array([xPos,yPos,zPos],dtype=np.float64)
        else:
            return None
        
    def wrap_to_pi(self,ang):
        return (ang+np.pi) % (2*np.pi) - np.pi
    
    def UAS1_ImuCallback(self,msg):
        self.UAS1_ImuMsg = msg
    
    def UAS1_GPSCallback(self,msg):
        self.UAS1_GPSMsg = msg

    def UAS1_OdomCallback(self,msg):
        self.UAS1_OdomMsg = msg

    def UAS2_ImuCallback(self,msg):
        self.UAS2_ImuMsg = msg

    def UAS2_GPSCallback(self,msg):
        self.UAS2_GPSMsg = msg

    def UAS2_OdomCallback(self,msg):
        self.UAS2_OdomMsg = msg

    def UAS3_ImuCallback(self,msg):
        self.UAS3_ImuMsg = msg

    def UAS3_GPSCallback(self,msg):
        self.UAS3_GPSMsg = msg

    def UAS3_OdomCallback(self,msg):
        self.UAS3_OdomMsg = msg

    def lm1_to_UAS1_callback(self,msg):
        self.lm1_to_UAS1_msg = msg

    def lm2_to_UAS1_callback(self,msg):
        self.lm2_to_UAS1_msg = msg

    def lm3_to_UAS1_callback(self,msg):
        self.lm3_to_UAS1_msg = msg

    def lm4_to_UAS1_callback(self,msg):
        self.lm4_to_UAS1_msg = msg

    def lm5_to_UAS1_callback(self,msg):
        self.lm5_to_UAS1_msg = msg

    def lm6_to_UAS1_callback(self,msg):
        self.lm6_to_UAS1_msg = msg

    def lm1_to_UAS2_callback(self,msg):
        self.lm1_to_UAS2_msg = msg

    def lm2_to_UAS2_callback(self,msg):
        self.lm2_to_UAS2_msg = msg

    def lm3_to_UAS2_callback(self,msg):
        self.lm3_to_UAS2_msg = msg

    def lm4_to_UAS2_callback(self,msg):
        self.lm4_to_UAS2_msg = msg

    def lm5_to_UAS2_callback(self,msg):
        self.lm5_to_UAS2_msg = msg

    def lm6_to_UAS2_callback(self,msg):
        self.lm6_to_UAS2_msg = msg

    def lm1_to_UAS3_callback(self,msg):
        self.lm1_to_UAS3_msg = msg

    def lm2_to_UAS3_callback(self,msg):
        self.lm2_to_UAS3_msg = msg

    def lm3_to_UAS3_callback(self,msg):
        self.lm3_to_UAS3_msg = msg

    def lm4_to_UAS3_callback(self,msg):
        self.lm4_to_UAS3_msg = msg

    def lm5_to_UAS3_callback(self,msg):
        self.lm5_to_UAS3_msg = msg

    def lm6_to_UAS3_callback(self,msg):
        self.lm6_to_UAS3_msg = msg

    def UAS2_to_UAS1_callback(self,msg):
        self.UAS2_to_UAS1_msg = msg

    def UAS3_to_UAS1_callback(self,msg):
        self.UAS3_to_UAS1_msg = msg

    def UAS1_to_UAS2_callback(self,msg):
        self.UAS1_to_UAS2_msg = msg

    def UAS3_to_UAS2_callback(self,msg):
        self.UAS3_to_UAS2_msg = msg

    def UAS1_to_UAS3_callback(self,msg):
        self.UAS1_to_UAS3_msg = msg

    def UAS2_to_UAS3_callback(self,msg):
        self.UAS2_to_UAS3_msg = msg

    def prediction(self,Ts):
        for i in range(self.steps):
            id              = 0
            ns              = self.NumStates
            nv              = self.NumVeh
            UAS1_Psi       = self.xHat[ns*id+3,0]
            UAS1_Vel       = self.xHat[ns*id+4:ns*id+7,:]
            UAS1_u        = UAS1_Vel[0,0]
            UAS1_v        = UAS1_Vel[1,0]
            UAS1_w        = UAS1_Vel[2,0]
            UAS1_Phi       = self.UAS1_Roll
            UAS1_Theta     = self.UAS1_Pitch
            UAS1_b_ax     = self.xHat[ns*id+7,0]
            UAS1_b_ay     = self.xHat[ns*id+8,0]
            UAS1_b_az     = self.xHat[ns*id+9,0]
            UAS1_R_mat    = np.array([[cos(UAS1_Theta)*cos(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Theta)*cos(UAS1_Psi)-cos(UAS1_Phi)*sin(UAS1_Psi), cos(UAS1_Phi)*sin(UAS1_Theta)*cos(UAS1_Psi)+sin(UAS1_Phi)*sin(UAS1_Psi)],
                                        [cos(UAS1_Theta)*sin(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Theta)*sin(UAS1_Psi)+cos(UAS1_Phi)*cos(UAS1_Psi), cos(UAS1_Phi)*sin(UAS1_Theta)*sin(UAS1_Psi)-sin(UAS1_Phi)*cos(UAS1_Psi)],
                                        [              -sin(UAS1_Theta),                                              sin(UAS1_Phi)*cos(UAS1_Theta),                                              cos(UAS1_Phi)*cos(UAS1_Theta)]],dtype=np.float64)
            UAS1_pos_dot = np.matmul(UAS1_R_mat,UAS1_Vel)
            u_dot   = self.UAS1_r*UAS1_v - self.UAS1_q*UAS1_w + self.UAS1_acc_u - UAS1_b_ax
            v_dot   = self.UAS1_p*UAS1_w - self.UAS1_r*UAS1_u + self.UAS1_acc_v - UAS1_b_ay
            w_dot   = self.UAS1_q*UAS1_u - self.UAS1_p*UAS1_v + self.UAS1_acc_w  - UAS1_b_az#- self.gravity
            UAS1_vel_dot = np.array([[u_dot],[v_dot],[w_dot]],dtype=np.float64)
            UAS1_psi_dot = np.array([[self.UAS1_q*(sin(UAS1_Phi)/cos(UAS1_Theta))+self.UAS1_r*(cos(UAS1_Phi)/cos(UAS1_Theta))]],dtype=np.float64)
            UAS1_bias_dot = np.zeros((10,1),dtype=np.float64)
            UAS1_At = np.array([[0, 0, 0, UAS1_w*(cos(UAS1_Psi)*sin(UAS1_Phi) - cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - UAS1_v*(cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - UAS1_u*cos(UAS1_Theta)*sin(UAS1_Psi), cos(UAS1_Psi)*cos(UAS1_Theta), cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta) - cos(UAS1_Phi)*sin(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, UAS1_w*(sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta)) - UAS1_v*(cos(UAS1_Phi)*sin(UAS1_Psi) - cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta)) + UAS1_u*cos(UAS1_Psi)*cos(UAS1_Theta), cos(UAS1_Theta)*sin(UAS1_Psi), cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta), cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta) - cos(UAS1_Psi)*sin(UAS1_Phi),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,               -sin(UAS1_Theta),                                                cos(UAS1_Theta)*sin(UAS1_Phi),                                                cos(UAS1_Phi)*cos(UAS1_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                  self.UAS1_r,                                                                 -self.UAS1_q, -1,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                  -self.UAS1_r,                                                                              0,                                                                  self.UAS1_p,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                   self.UAS1_q,                                                                 -self.UAS1_p,                                                                              0,  0,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0,                                                                                                                                                                                                                                0,                               0,                                                                              0,                                                                              0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float64)
            UAS1_Bt = np.array([[0, 0, 0,   UAS1_v*(sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta)) + UAS1_w*(cos(UAS1_Phi)*sin(UAS1_Psi) - cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta)), UAS1_w*cos(UAS1_Phi)*cos(UAS1_Psi)*cos(UAS1_Theta) - UAS1_u*cos(UAS1_Psi)*sin(UAS1_Theta) + UAS1_v*cos(UAS1_Psi)*cos(UAS1_Theta)*sin(UAS1_Phi),         0,                               0,                               0],
                                 [0, 0, 0, - UAS1_v*(cos(UAS1_Psi)*sin(UAS1_Phi) - cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - UAS1_w*(cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)), UAS1_w*cos(UAS1_Phi)*cos(UAS1_Theta)*sin(UAS1_Psi) - UAS1_u*sin(UAS1_Psi)*sin(UAS1_Theta) + UAS1_v*cos(UAS1_Theta)*sin(UAS1_Phi)*sin(UAS1_Psi),         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                     UAS1_v*cos(UAS1_Phi)*cos(UAS1_Theta) - UAS1_w*cos(UAS1_Theta)*sin(UAS1_Phi),                                            - UAS1_u*cos(UAS1_Theta) - UAS1_w*cos(UAS1_Phi)*sin(UAS1_Theta) - UAS1_v*sin(UAS1_Phi)*sin(UAS1_Theta),         0,                               0,                               0],
                                 [0, 0, 0,                                                                                       (self.UAS1_q*cos(UAS1_Phi))/cos(UAS1_Theta) - (self.UAS1_r*sin(UAS1_Phi))/cos(UAS1_Theta),                    (self.UAS1_r*cos(UAS1_Phi)*sin(UAS1_Theta))/cos(UAS1_Theta)**2 + (self.UAS1_q*sin(UAS1_Phi)*sin(UAS1_Theta))/cos(UAS1_Theta)**2,         0, sin(UAS1_Phi)/cos(UAS1_Theta), cos(UAS1_Phi)/cos(UAS1_Theta)],
                                 [1, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                       -UAS1_w,                        UAS1_v],
                                 [0, 1, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,  UAS1_w,                               0,                       -UAS1_u],
                                 [0, 0, 1,                                                                                                                                                                                       0,                                                                                                                                                            0, -UAS1_v,                        UAS1_u,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0],
                                 [0, 0, 0,                                                                                                                                                                                       0,                                                                                                                                                            0,         0,                               0,                               0]],dtype=np.float64)
            id               = 1
            UAS2_Psi       = self.xHat[ns*id+3,0]
            UAS2_Vel       = self.xHat[ns*id+4:ns*id+7,:]
            UAS2_u        = UAS2_Vel[0,0]
            UAS2_v        = UAS2_Vel[1,0]
            UAS2_w        = UAS2_Vel[2,0]
            UAS2_Phi       = self.UAS2_Roll
            UAS2_Theta     = self.UAS2_Pitch
            UAS2_b_ax     = self.xHat[ns*id+7,0]
            UAS2_b_ay     = self.xHat[ns*id+8,0]
            UAS2_b_az     = self.xHat[ns*id+9,0]
            UAS2_R_mat    = np.array([[cos(UAS2_Theta)*cos(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Theta)*cos(UAS2_Psi)-cos(UAS2_Phi)*sin(UAS2_Psi), cos(UAS2_Phi)*sin(UAS2_Theta)*cos(UAS2_Psi)+sin(UAS2_Phi)*sin(UAS2_Psi)],
                                         [cos(UAS2_Theta)*sin(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Theta)*sin(UAS2_Psi)+cos(UAS2_Phi)*cos(UAS2_Psi), cos(UAS2_Phi)*sin(UAS2_Theta)*sin(UAS2_Psi)-sin(UAS2_Phi)*cos(UAS2_Psi)],
                                         [              -sin(UAS2_Theta),                                              sin(UAS2_Phi)*cos(UAS2_Theta),                                              cos(UAS2_Phi)*cos(UAS2_Theta)]],dtype=np.float64)
            UAS2_pos_dot = np.matmul(UAS2_R_mat,UAS2_Vel)
            u_dot   = self.UAS2_r*UAS2_v - self.UAS2_q*UAS2_w + self.UAS2_acc_u - UAS2_b_ax
            v_dot   = self.UAS2_p*UAS2_w - self.UAS2_r*UAS2_u + self.UAS2_acc_v - UAS2_b_ay
            w_dot   = self.UAS2_q*UAS2_u - self.UAS2_p*UAS2_v + self.UAS2_acc_w  - UAS2_b_az#- self.gravity
            UAS2_vel_dot = np.array([[u_dot],[v_dot],[w_dot]],dtype=np.float64)
            UAS2_psi_dot = np.array([[self.UAS2_q*(sin(UAS2_Phi)/cos(UAS2_Theta))+self.UAS2_r*(cos(UAS2_Phi)/cos(UAS2_Theta))]],dtype=np.float64)
            UAS2_bias_dot = np.zeros((10,1),dtype=np.float64)
            UAS2_At = np.array([[0, 0, 0, UAS2_w*(cos(UAS2_Psi)*sin(UAS2_Phi) - cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - UAS2_v*(cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - UAS2_u*cos(UAS2_Theta)*sin(UAS2_Psi), cos(UAS2_Psi)*cos(UAS2_Theta), cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta) - cos(UAS2_Phi)*sin(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, UAS2_w*(sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta)) - UAS2_v*(cos(UAS2_Phi)*sin(UAS2_Psi) - cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta)) + UAS2_u*cos(UAS2_Psi)*cos(UAS2_Theta), cos(UAS2_Theta)*sin(UAS2_Psi), cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta), cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta) - cos(UAS2_Psi)*sin(UAS2_Phi),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                -sin(UAS2_Theta),                                                   cos(UAS2_Theta)*sin(UAS2_Phi),                                                   cos(UAS2_Phi)*cos(UAS2_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                      self.UAS2_r,                                                                     -self.UAS2_q, -1,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                   -self.UAS2_r,                                                                                   0,                                                                      self.UAS2_p,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                    self.UAS2_q,                                                                     -self.UAS2_p,                                                                                   0,  0,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float64)
            UAS2_Bt = np.array([[0, 0, 0,   UAS2_v*(sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta)) + UAS2_w*(cos(UAS2_Phi)*sin(UAS2_Psi) - cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta)), UAS2_w*cos(UAS2_Phi)*cos(UAS2_Psi)*cos(UAS2_Theta) - UAS2_u*cos(UAS2_Psi)*sin(UAS2_Theta) + UAS2_v*cos(UAS2_Psi)*cos(UAS2_Theta)*sin(UAS2_Phi),          0,                                 0,                                 0],
                                  [0, 0, 0, - UAS2_v*(cos(UAS2_Psi)*sin(UAS2_Phi) - cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - UAS2_w*(cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)), UAS2_w*cos(UAS2_Phi)*cos(UAS2_Theta)*sin(UAS2_Psi) - UAS2_u*sin(UAS2_Psi)*sin(UAS2_Theta) + UAS2_v*cos(UAS2_Theta)*sin(UAS2_Phi)*sin(UAS2_Psi),          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                           UAS2_v*cos(UAS2_Phi)*cos(UAS2_Theta) - UAS2_w*cos(UAS2_Theta)*sin(UAS2_Phi),                                               - UAS2_u*cos(UAS2_Theta) - UAS2_w*cos(UAS2_Phi)*sin(UAS2_Theta) - UAS2_v*sin(UAS2_Phi)*sin(UAS2_Theta),          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                             (self.UAS2_q*cos(UAS2_Phi))/cos(UAS2_Theta) - (self.UAS2_r*sin(UAS2_Phi))/cos(UAS2_Theta),                       (self.UAS2_r*cos(UAS2_Phi)*sin(UAS2_Theta))/cos(UAS2_Theta)**2 + (self.UAS2_q*sin(UAS2_Phi)*sin(UAS2_Theta))/cos(UAS2_Theta)**2,          0, sin(UAS2_Phi)/cos(UAS2_Theta), cos(UAS2_Phi)/cos(UAS2_Theta)],
                                  [1, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                        -UAS2_w,                         UAS2_v],
                                  [0, 1, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,  UAS2_w,                                 0,                        -UAS2_u],
                                  [0, 0, 1,                                                                                                                                                                                                   0,                                                                                                                                                                       0, -UAS2_v,                         UAS2_u,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0]],dtype=np.float64)
            id               = 2
            UAS3_Psi       = self.xHat[ns*id+3,0]
            UAS3_Vel       = self.xHat[ns*id+4:ns*id+7,:]
            UAS3_u        = UAS3_Vel[0,0]
            UAS3_v        = UAS3_Vel[1,0]
            UAS3_w        = UAS3_Vel[2,0]
            UAS3_Phi       = self.UAS3_Roll
            UAS3_Theta     = self.UAS3_Pitch
            UAS3_b_ax     = self.xHat[ns*id+7,0]
            UAS3_b_ay     = self.xHat[ns*id+8,0]
            UAS3_b_az     = self.xHat[ns*id+9,0]
            UAS3_R_mat    = np.array([[cos(UAS3_Theta)*cos(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Theta)*cos(UAS3_Psi)-cos(UAS3_Phi)*sin(UAS3_Psi), cos(UAS3_Phi)*sin(UAS3_Theta)*cos(UAS3_Psi)+sin(UAS3_Phi)*sin(UAS3_Psi)],
                                         [cos(UAS3_Theta)*sin(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Theta)*sin(UAS3_Psi)+cos(UAS3_Phi)*cos(UAS3_Psi), cos(UAS3_Phi)*sin(UAS3_Theta)*sin(UAS3_Psi)-sin(UAS3_Phi)*cos(UAS3_Psi)],
                                         [               -sin(UAS3_Theta),                                                 sin(UAS3_Phi)*cos(UAS3_Theta),                                                 cos(UAS3_Phi)*cos(UAS3_Theta)]],dtype=np.float64)
            UAS3_pos_dot = np.matmul(UAS3_R_mat,UAS3_Vel)
            u_dot   = self.UAS3_r*UAS3_v - self.UAS3_q*UAS3_w + self.UAS3_acc_u - UAS3_b_ax
            v_dot   = self.UAS3_p*UAS3_w - self.UAS3_r*UAS3_u + self.UAS3_acc_v - UAS3_b_ay
            w_dot   = self.UAS3_q*UAS3_u - self.UAS3_p*UAS3_v + self.UAS3_acc_w  - UAS3_b_az#- self.gravity
            UAS3_vel_dot = np.array([[u_dot],[v_dot],[w_dot]],dtype=np.float64)
            UAS3_psi_dot = np.array([[self.UAS3_q*(sin(UAS3_Phi)/cos(UAS3_Theta))+self.UAS3_r*(cos(UAS3_Phi)/cos(UAS3_Theta))]],dtype=np.float64)
            UAS3_bias_dot = np.zeros((10,1),dtype=np.float64)
            UAS3_At = np.array([[0, 0, 0, UAS3_w*(cos(UAS3_Psi)*sin(UAS3_Phi) - cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - UAS3_v*(cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - UAS3_u*cos(UAS3_Theta)*sin(UAS3_Psi), cos(UAS3_Psi)*cos(UAS3_Theta), cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta) - cos(UAS3_Phi)*sin(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, UAS3_w*(sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta)) - UAS3_v*(cos(UAS3_Phi)*sin(UAS3_Psi) - cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta)) + UAS3_u*cos(UAS3_Psi)*cos(UAS3_Theta), cos(UAS3_Theta)*sin(UAS3_Psi), cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta), cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta) - cos(UAS3_Psi)*sin(UAS3_Phi),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                -sin(UAS3_Theta),                                                   cos(UAS3_Theta)*sin(UAS3_Phi),                                                   cos(UAS3_Phi)*cos(UAS3_Theta),  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                      self.UAS3_r,                                                                     -self.UAS3_q, -1,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                   -self.UAS3_r,                                                                                   0,                                                                      self.UAS3_p,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                    self.UAS3_q,                                                                     -self.UAS3_p,                                                                                   0,  0,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0,                                                                                                                                                                                                                                               0,                                 0,                                                                                   0,                                                                                   0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float64)
            UAS3_Bt = np.array([[0, 0, 0,   UAS3_v*(sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta)) + UAS3_w*(cos(UAS3_Phi)*sin(UAS3_Psi) - cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta)), UAS3_w*cos(UAS3_Phi)*cos(UAS3_Psi)*cos(UAS3_Theta) - UAS3_u*cos(UAS3_Psi)*sin(UAS3_Theta) + UAS3_v*cos(UAS3_Psi)*cos(UAS3_Theta)*sin(UAS3_Phi),          0,                                 0,                                 0],
                                  [0, 0, 0, - UAS3_v*(cos(UAS3_Psi)*sin(UAS3_Phi) - cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - UAS3_w*(cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)), UAS3_w*cos(UAS3_Phi)*cos(UAS3_Theta)*sin(UAS3_Psi) - UAS3_u*sin(UAS3_Psi)*sin(UAS3_Theta) + UAS3_v*cos(UAS3_Theta)*sin(UAS3_Phi)*sin(UAS3_Psi),          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                           UAS3_v*cos(UAS3_Phi)*cos(UAS3_Theta) - UAS3_w*cos(UAS3_Theta)*sin(UAS3_Phi),                                               - UAS3_u*cos(UAS3_Theta) - UAS3_w*cos(UAS3_Phi)*sin(UAS3_Theta) - UAS3_v*sin(UAS3_Phi)*sin(UAS3_Theta),          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                             (self.UAS3_q*cos(UAS3_Phi))/cos(UAS3_Theta) - (self.UAS3_r*sin(UAS3_Phi))/cos(UAS3_Theta),                       (self.UAS3_r*cos(UAS3_Phi)*sin(UAS3_Theta))/cos(UAS3_Theta)**2 + (self.UAS3_q*sin(UAS3_Phi)*sin(UAS3_Theta))/cos(UAS3_Theta)**2,          0, sin(UAS3_Phi)/cos(UAS3_Theta), cos(UAS3_Phi)/cos(UAS3_Theta)],
                                  [1, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                        -UAS3_w,                         UAS3_v],
                                  [0, 1, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,  UAS3_w,                                 0,                        -UAS3_u],
                                  [0, 0, 1,                                                                                                                                                                                                   0,                                                                                                                                                                       0, -UAS3_v,                         UAS3_u,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0],
                                  [0, 0, 0,                                                                                                                                                                                                   0,                                                                                                                                                                       0,          0,                                 0,                                 0]],dtype=np.float64)
            At = np.zeros((ns*3,ns*3),dtype=np.float64)
            At[ns*0+0:ns*0+ns,ns*0+0:ns*0+ns] = UAS1_At[0:17,0:17]
            At[ns*1+0:ns*1+ns,ns*1+0:ns*1+ns] = UAS2_At[0:17,0:17]
            At[ns*2+0:ns*2+ns,ns*2+0:ns*2+ns] = UAS3_At[0:17,0:17]
            Bt = np.zeros((3*ns,24),dtype=np.float64)
            Bt[ns*0+0:ns*0+ns,0:8] = UAS1_Bt[0:ns,:]
            Bt[ns*1+0:ns*1+ns,8:16] = UAS2_Bt[0:ns,:]
            Bt[ns*2+0:ns*2+ns,16:24] = UAS3_Bt[0:ns,:]
            QU = np.zeros((24,24),dtype=np.float64)
            qu = np.array([[self.axSigma**2,               0,               0,                0,                  0,                0,                0,                0],
                           [              0, self.aySigma**2,               0,                0,                  0,                0,                0,                0],
                           [              0,               0, self.azSigma**2,                0,                  0,                0,                0,                0],
                           [              0,               0,               0, self.phiSigma**2,                  0,                0,                0,                0],
                           [              0,               0,               0,                0, self.thetaSigma**2,                0,                0,                0],
                           [              0,               0,               0,                0,                  0, self.omgSigma**2,                0,                0],
                           [              0,               0,               0,                0,                  0,                0, self.omgSigma**2,                0],
                           [              0,               0,               0,                0,                  0,                0,                0, self.omgSigma**2]], dtype=np.float64)
            QU[0:8,0:8] = qu
            QU[8:16,8:16] = qu
            QU[16:24,16:24] = qu
            Qt = np.matmul(Bt,np.matmul(QU,np.transpose(Bt)))
            id = 0
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.4
            Qt[ns*id+10,ns*id+10] = 0.4
            Qt[ns*id+11,ns*id+11] = 0.4
            Qt[ns*id+12,ns*id+12] = 0.4
            Qt[ns*id+13,ns*id+13] = 0.4
            Qt[ns*id+14,ns*id+14] = 0.4
            Qt[ns*id+15,ns*id+15] = 0.4
            Qt[ns*id+16,ns*id+16] = 0.8
            #Qt[ns*id+17,ns*id+17] = 0.8
            id = 1
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.4
            Qt[ns*id+10,ns*id+10] = 0.8
            Qt[ns*id+11,ns*id+11] = 0.8
            Qt[ns*id+12,ns*id+12] = 0.8
            Qt[ns*id+13,ns*id+13] = 0.8
            Qt[ns*id+14,ns*id+14] = 0.8
            Qt[ns*id+15,ns*id+15] = 0.8
            Qt[ns*id+16,ns*id+16] = 0.8
            #Qt[ns*id+17,ns*id+17] = 0.8
            id = 2
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.4
            Qt[ns*id+10,ns*id+10] = 0.8
            Qt[ns*id+11,ns*id+11] = 0.8
            Qt[ns*id+12,ns*id+12] = 0.8
            Qt[ns*id+13,ns*id+13] = 0.8
            Qt[ns*id+14,ns*id+14] = 0.8
            Qt[ns*id+15,ns*id+15] = 0.8
            Qt[ns*id+16,ns*id+16] = 0.8
            #Qt[ns*id+17,ns*id+17] = 0.8
            self.xHat = self.xHat + (Ts/self.steps)*np.concatenate((UAS1_pos_dot,UAS1_psi_dot,UAS1_vel_dot,UAS1_bias_dot,
                                                                    UAS2_pos_dot,UAS2_psi_dot,UAS2_vel_dot,UAS2_bias_dot,
                                                                    UAS3_pos_dot,UAS3_psi_dot,UAS3_vel_dot,UAS3_bias_dot),axis=0)
            self.pCov   = self.pCov + (Ts/self.steps)*(np.matmul(At,self.pCov) + np.matmul(self.pCov,np.transpose(At)) + Qt)

    def gpsMeasurementUpdate(self,veh,msg):
        gpsTime = (msg.header.stamp).to_sec()
        ns = self.NumStates
        if veh == 'UAS1':
            odomTime = self.UAS1_Time
            id       = 0
            val      = 0
        if veh == 'UAS2':
            odomTime = self.UAS2_Time
            id       = 1
            val      = 0
        if veh == 'UAS3':
            odomTime = self.UAS3_Time
            val      = 0
            id       = 2
        
        timeDiff = abs(gpsTime-odomTime)
        if timeDiff < 0.2:
            lat              = float(msg.latitude)
            lon              = float(msg.longitude)
            alt              = float(msg.altitude)
            gpsPos           = utm.from_latlon(lat,lon)
            xPos, yPos, zPos = self.convert_to_local(gpsPos[0],gpsPos[1],alt)
            zt               = np.array([[xPos],[yPos],[zPos+val]],dtype=np.float64)
            Ht               = np.zeros((3,3*ns),dtype=np.float64)
            Ht[0][ns*id+0]   = 1.0
            Ht[1][ns*id+1]   = 1.0
            Ht[2][ns*id+2]   = 1.0
            Rt               = np.array([[self.gpsSigma**2, 0, 0],
                                         [0, self.gpsSigma**2, 0],
                                         [0, 0, self.gpsSigma**2]],dtype=np.float64)
            mat              = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
            inv_mat          = np.linalg.inv(mat)
            Lt               = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
            self.pCov        = np.dot(np.identity(3*ns)-np.dot(Lt,Ht),self.pCov)
            self.xHat        = np.add(self.xHat,np.dot(Lt,zt-np.dot(Ht,self.xHat)))

    def heightMeasurementUpdate(self,veh,msg):
        ns = self.NumStates
        nv = self.NumVeh
        gpsTime = (msg.header.stamp).to_sec()
        if veh == 'UAS1':
            odomTime = self.UAS1_Time
            id       = 0
            val      = 0
        if veh == 'UAS2':
            odomTime = self.UAS2_Time
            id       = 1
            val      = 0
        if veh == 'UAS3':
            odomTime = self.UAS3_Time
            val      = 0
            id       = 2
        
        timeDiff = abs(gpsTime-odomTime)
        if timeDiff < 0.1:
            lat              = float(msg.latitude)
            lon              = float(msg.longitude)
            alt              = float(msg.altitude)
            gpsPos           = utm.from_latlon(lat,lon)
            xPos, yPos, zPos = self.convert_to_local(gpsPos[0],gpsPos[1],alt)
            zt               = np.array([[zPos+val]],dtype=np.float64)
            Ht               = np.zeros((1,3*ns),dtype=np.float64)
            Ht[0][ns*id+2]   = 1.0
            Rt               = np.array([[self.gpsSigma**2]],dtype=np.float64)
            mat              = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
            inv_mat          = np.linalg.inv(mat)
            Lt               = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
            self.pCov        = np.dot(np.identity(3*ns)-np.dot(Lt,Ht),self.pCov)
            self.xHat        = np.add(self.xHat,np.dot(Lt,zt-np.dot(Ht,self.xHat)))

    def velocityMeasurementUpdate(self,veh,msg):
        ns = self.NumStates
        odomTime = (msg.header.stamp).to_sec()
        if veh == 'UAS1':
            accelTime = self.UAS1_Time
            id       = 0
        if veh == 'UAS2':
            accelTime = self.UAS2_Time
            id       = 1
        if veh == 'UAS3':
            accelTime = self.UAS3_Time
            id       = 2
        timeDiff = abs(accelTime-odomTime)
        if timeDiff < 0.1:
            zt = np.array([[msg.twist.twist.linear.x],[msg.twist.twist.linear.y],[msg.twist.twist.linear.z]],dtype=np.float64)
            zHat = np.array([[self.xHat[ns*id+4,0]],[self.xHat[ns*id+5,0]],[self.xHat[ns*id+6,0]]],dtype=np.float64)
            Rt = np.array([[self.velSigma**2,0,0],[0,self.velSigma**2,0],[0,0,self.velSigma**2]],dtype=np.float64)
            Ht = np.zeros((3,ns*3),dtype=np.float64)
            Ht[0,ns*id+4] = 1.0
            Ht[1,ns*id+5] = 1.0
            Ht[2,ns*id+6] = 1.0

            mat              = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
            inv_mat          = np.linalg.inv(mat)
            Lt               = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
            self.pCov        = np.dot(np.identity(3*ns)-np.dot(Lt,Ht),self.pCov)
            self.xHat        = np.add(self.xHat,np.dot(Lt,zt-zHat))

    def landmarkRangeMeasurementUpdate(self,veh,msg,lm,lmID):
        if msg.range > 2.5:
            ns = self.NumStates
            lmX      = lm[0]
            lmY      = lm[1]
            lmZ      = lm[2]
            Ht       = np.zeros((1,3*ns),dtype=np.float64)
            zt       = np.array([[msg.range/1000.0]],dtype=np.float64)
            
            rhoTime  = (msg.header.stamp).to_sec()

            if veh == 'UAS1':
                odomTime = self.UAS1_Time
                id       = 0
            if veh == 'UAS2':
                odomTime = self.UAS2_Time
                id       = 1
            if veh == 'UAS3':
                odomTime = self.UAS3_Time
                id       = 2

            if lmID == 'lm1':
                bid = 10
            if lmID == 'lm2':
                bid = 11
            if lmID == 'lm3':
                bid = 12
            if lmID == 'lm4':
                bid = 13
            if lmID == 'lm5':
                bid = 14
            if lmID == 'lm6':
                bid = 15

            xPos            = self.xHat[ns*id+0,0]
            yPos            = self.xHat[ns*id+1,0]
            zPos            = self.xHat[ns*id+2,0]
            bias            = self.xHat[ns*id+bid,0]

            xDiff           = xPos-lmX
            yDiff           = yPos-lmY
            zDiff           = zPos-lmZ
            zHat            = np.array([[((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)+bias]],dtype=np.float64)
            rhoHat          = ((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)
            Ht[0,ns*id+0]   = xDiff/rhoHat
            Ht[0,ns*id+1]   = yDiff/rhoHat
            Ht[0,ns*id+2]   = zDiff/rhoHat
            Ht[0,ns*id+bid] = 1.0

            timeDiff = abs(rhoTime-odomTime)
            if timeDiff < self.rhoTimeDiff:
                Rt        = np.array([[self.rhoSigma**2]],dtype=np.float64)
                mat       = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
                inv_mat   = np.linalg.inv(mat)
                Lt        = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
                self.pCov = np.dot(np.identity(3*ns)-np.dot(Lt,Ht),self.pCov)
                self.xHat = np.add(self.xHat,np.dot(Lt,zt-zHat))

                print("Measurement update with range between",veh,"and",lmID)

    def intervehicleRangeMeasurementUpdate(self,veh1,veh2,msg):
        if msg.range > 2.5:
            ns      = self.NumStates
            nv      = self.NumVeh
            Ht      = np.zeros((1,3*ns),dtype=np.float64)
            zt      = np.array([[msg.range/1000.0]],dtype=np.float64)
            rhoTime = (msg.header.stamp).to_sec()

            if veh1 == 'UAS1':
                odomTime = self.UAS1_Time
                id1      = 0
                if veh2 == 'UAS2':
                    id2 = 1
                    bid = 15
                if veh2 == 'UAS3':
                    id2 = 2
                    bid = 16
            if veh1 == 'UAS2':
                odomTime = self.UAS2_Time
                id1      = 1
                if veh2 == 'UAS1':
                    id2 = 0
                    bid = 15
                if veh2 == 'UAS3':
                    id2 = 2
                    bid = 16
            if veh1 == 'UAS3':
                odomTime = self.UAS3_Time
                id1      = 2
                if veh2 == 'UAS1':
                    id2 = 0
                    bid = 15
                if veh2 == 'UAS2':
                    id2 = 1
                    bid = 16
            
            x1_pos           = self.xHat[ns*id1+0,0]
            y1_pos           = self.xHat[ns*id1+1,0]
            z1_pos           = self.xHat[ns*id1+2,0]

            x2_pos           = self.xHat[ns*id2+0,0]
            y2_pos           = self.xHat[ns*id2+1,0]
            z2_pos           = self.xHat[ns*id2+2,0]

            bias             = self.xHat[ns*id1+bid,0]

            xDiff            = x1_pos - x2_pos
            yDiff            = y1_pos - y2_pos
            zDiff            = z1_pos - z2_pos

            zHat             = np.array([[((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)+bias]],dtype=np.float64)
            rhoHat           = ((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)
            Ht[0,ns*id1+0]   = xDiff/rhoHat
            Ht[0,ns*id1+1]   = yDiff/rhoHat
            Ht[0,ns*id1+2]   = zDiff/rhoHat
            Ht[0,ns*id2+0]   = -xDiff/rhoHat
            Ht[0,ns*id2+1]   = -yDiff/rhoHat
            Ht[0,ns*id2+2]   = -zDiff/rhoHat
            Ht[0,ns*id1+bid] = 1.0

            timeDiff         = abs(rhoTime-odomTime)
            if timeDiff < self.rhoTimeDiff:
                Rt        = np.array([[self.rhoSigma**2]],dtype=np.float64)
                mat       = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
                inv_mat   = np.linalg.inv(mat)
                Lt        = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
                self.pCov = np.dot(np.identity(3*ns)-np.dot(Lt,Ht),self.pCov)
                self.xHat = np.add(self.xHat,np.dot(Lt,zt-zHat))

                print("Measurement update with range between",veh1,"and",veh2)


    def calculate_error(self,msg,vel,yaw,veh):
        ns = self.NumStates
        if veh == "UAS1":
            id  = 0
            val = 0
        if veh == "UAS2":
            id  = 1
            val = 0
        if veh == "UAS3":
            id  = 2
            val = 0
        lat              = float(msg.latitude)
        lon              = float(msg.longitude)
        alt              = float(msg.altitude)
        pos              = utm.from_latlon(lat,lon)
        xPos, yPos, zPos = self.convert_to_local(pos[0],pos[1],alt)
        local_pos        = np.array([xPos,yPos,zPos],dtype=np.float64) 
        u                = vel[0]
        v                = vel[1]
        w                = vel[2]
        psi              = yaw

        xEst             = self.xHat[ns*id+0,0]
        yEst             = self.xHat[ns*id+1,0]
        zEst             = self.xHat[ns*id+2,0]
        psiEst           = self.xHat[ns*id+3,0]
        uEst             = self.xHat[ns*id+4,0]
        vEst             = self.xHat[ns*id+5,0]
        wEst             = self.xHat[ns*id+6,0]
        b_ax_est         = self.xHat[ns*id+7,0]
        b_ay_est         = self.xHat[ns*id+8,0]
        b_az_est         = self.xHat[ns*id+9,0]
        b_rho_lm1        = self.xHat[ns*id+10,0]
        b_rho_lm2        = self.xHat[ns*id+11,0]
        b_rho_lm3        = self.xHat[ns*id+12,0]
        b_rho_lm4        = self.xHat[ns*id+13,0]
        b_rho_lm5        = self.xHat[ns*id+14,0]
        #b_rho_lm6        = self.xHat[ns*id+15,0]
        b_rho_v1         = self.xHat[ns*id+15,0]
        b_rho_v2         = self.xHat[ns*id+16,0]

        err              = np.array([xPos-xEst,yPos-yEst,zPos+val-zEst,(180.0/np.pi)*self.wrap_to_pi(psi-psiEst),u-uEst,v-vEst,w-wEst,
                                     0-b_ax_est,0-b_ay_est,0-b_az_est,
                                     0-b_rho_lm1,0-b_rho_lm2,0-b_rho_lm3,0-b_rho_lm4,0-b_rho_lm5,
                                     0-b_rho_v1, 0-b_rho_v2],dtype=np.float64)
        
        sigma            = np.array([(self.pCov[ns*id+0,ns*id+0])**(0.5),
                                     (self.pCov[ns*id+1,ns*id+1])**(0.5),
                                     (self.pCov[ns*id+2,ns*id+2])**(0.5),
                                     (180.0/np.pi)*(self.pCov[ns*id+3,ns*id+3])**(0.5),
                                     (self.pCov[ns*id+4,ns*id+4])**(0.5),
                                     (self.pCov[ns*id+5,ns*id+5])**(0.5),
                                     (self.pCov[ns*id+6,ns*id+6])**(0.5),
                                     (self.pCov[ns*id+7,ns*id+7])**(0.5),
                                     (self.pCov[ns*id+8,ns*id+8])**(0.5),
                                     (self.pCov[ns*id+9,ns*id+9])**(0.5),
                                     (self.pCov[ns*id+10,ns*id+10])**(0.5),
                                     (self.pCov[ns*id+11,ns*id+11])**(0.5),
                                     (self.pCov[ns*id+12,ns*id+12])**(0.5),
                                     (self.pCov[ns*id+13,ns*id+13])**(0.5),
                                     (self.pCov[ns*id+14,ns*id+14])**(0.5),
                                     (self.pCov[ns*id+15,ns*id+15])**(0.5),
                                     (self.pCov[ns*id+16,ns*id+16])**(0.5)],dtype=np.float64)
        pos_3sigma       = np.array([3*(self.pCov[ns*id+0,ns*id+0])**(0.5),
                                     3*(self.pCov[ns*id+1,ns*id+1])**(0.5),
                                     3*(self.pCov[ns*id+2,ns*id+2])**(0.5),
                                     (180.0/np.pi)*3*(self.pCov[ns*id+3,ns*id+3])**(0.5),
                                     3*(self.pCov[ns*id+4,ns*id+4])**(0.5),
                                     3*(self.pCov[ns*id+5,ns*id+5])**(0.5),
                                     3*(self.pCov[ns*id+6,ns*id+6])**(0.5),
                                     3*(self.pCov[ns*id+7,ns*id+7])**(0.5),
                                     3*(self.pCov[ns*id+8,ns*id+8])**(0.5),
                                     3*(self.pCov[ns*id+9,ns*id+9])**(0.5),
                                     3*(self.pCov[ns*id+10,ns*id+10])**(0.5),
                                     3*(self.pCov[ns*id+11,ns*id+11])**(0.5),
                                     3*(self.pCov[ns*id+12,ns*id+12])**(0.5),
                                     3*(self.pCov[ns*id+13,ns*id+13])**(0.5),
                                     3*(self.pCov[ns*id+14,ns*id+14])**(0.5),
                                     3*(self.pCov[ns*id+15,ns*id+15])**(0.5),
                                     3*(self.pCov[ns*id+16,ns*id+16])**(0.5)],dtype=np.float64)
        neg_3sigma       = -pos_3sigma
        return err, sigma, pos_3sigma, neg_3sigma, local_pos
    
    def publish_msg(self,veh,err,sigma,posSigma,negSigma,loc_pos):
        ns = self.NumStates
        if veh == "UAS1":
            id  = 0
            imuMsg = self.UAS1_ImuMsg
            gpsMsg = self.UAS1_GPSMsg
            odomMsg = self.UAS1_OdomMsg
            yaw = self.UAS1_Yaw
            time = self.UAS1_Time
            val  = 0
        if veh == "UAS2":
            id  = 1
            imuMsg = self.UAS2_ImuMsg
            gpsMsg = self.UAS2_GPSMsg
            odomMsg = self.UAS2_OdomMsg
            yaw = self.UAS2_Yaw
            time = self.UAS2_Time
            val = 0
        if veh == "UAS3":
            id  = 2
            imuMsg = self.UAS3_ImuMsg
            gpsMsg = self.UAS3_GPSMsg
            odomMsg = self.UAS3_OdomMsg
            yaw = self.UAS3_Yaw
            time = self.UAS3_Time
            val = 0

        estPub = localization_estimates()
        estPub.header = imuMsg.header
        estPub.header.stamp = rospy.Time.from_sec(time)
        estPub.pn        = self.xHat[ns*id+0,0]
        estPub.pe        = self.xHat[ns*id+1,0]
        estPub.pd        = self.xHat[ns*id+2,0]
        estPub.psi       = (180.0/np.pi)*self.wrap_to_pi(self.xHat[ns*id+3,0])
        estPub.u         = self.xHat[ns*id+4,0]
        estPub.v         = self.xHat[ns*id+5,0]
        estPub.w         = self.xHat[ns*id+6,0]
        estPub.b_ax      = self.xHat[ns*id+7,0]
        estPub.b_ay      = self.xHat[ns*id+8,0]
        estPub.b_az      = self.xHat[ns*id+9,0]
        estPub.b_rho_lm1 = self.xHat[ns*id+10,0]
        estPub.b_rho_lm2 = self.xHat[ns*id+11,0]
        estPub.b_rho_lm3 = self.xHat[ns*id+12,0]
        estPub.b_rho_lm4 = self.xHat[ns*id+13,0]
        estPub.b_rho_lm5 = self.xHat[ns*id+14,0]
        #estPub.b_rho_lm6 = self.xHat[ns*id+15,0]
        estPub.b_rho_v1  = self.xHat[ns*id+15,0]
        estPub.b_rho_v2  = self.xHat[ns*id+16,0]

        errPub = localization_error()
        errPub.header = imuMsg.header
        errPub.header.stamp = rospy.Time.from_sec(time)
        errPub.pn   = err[0]
        errPub.pe   = err[1]
        errPub.pd   = err[2]
        errPub.psi  = err[3]
        errPub.u    = err[4]
        errPub.v    = err[5]
        errPub.w    = err[6]
        errPub.b_ax = err[7]
        errPub.b_ay = err[8]
        errPub.b_az = err[9]
        errPub.b_rho_lm1 = err[10]
        errPub.b_rho_lm2 = err[11]
        errPub.b_rho_lm3 = err[12]
        errPub.b_rho_lm4 = err[13]
        errPub.b_rho_lm5 = err[14]
        #errPub.b_rho_lm6 = err[15]
        errPub.b_rho_v1  = err[15]
        errPub.b_rho_v2  = err[16]

        errPub.sigma_pn  = sigma[0]
        errPub.sigma_pe  = sigma[1]
        errPub.sigma_pd  = sigma[2]
        errPub.sigma_psi = sigma[3]
        errPub.sigma_u   = sigma[4]
        errPub.sigma_v   = sigma[5]
        errPub.sigma_w   = sigma[6]
        errPub.sigma_b_ax = sigma[7]
        errPub.sigma_b_ay = sigma[8]
        errPub.sigma_b_az = sigma[9]
        errPub.sigma_b_rho_lm1 = sigma[10]
        errPub.sigma_b_rho_lm2 = sigma[11]
        errPub.sigma_b_rho_lm3 = sigma[12]
        errPub.sigma_b_rho_lm4 = sigma[13]
        errPub.sigma_b_rho_lm5 = sigma[14]
        #errPub.sigma_b_rho_lm6 = sigma[15]
        errPub.sigma_b_rho_v1  = sigma[15]
        errPub.sigma_b_rho_v2  = sigma[16]

        errPub.pos_3_sigma_pn        = posSigma[0]
        errPub.pos_3_sigma_pe        = posSigma[1]
        errPub.pos_3_sigma_pd        = posSigma[2]
        errPub.pos_3_sigma_psi       = posSigma[3]
        errPub.pos_3_sigma_u         = posSigma[4]
        errPub.pos_3_sigma_v         = posSigma[5]
        errPub.pos_3_sigma_w         = posSigma[6]
        errPub.pos_3_sigma_b_ax      = posSigma[7]
        errPub.pos_3_sigma_b_ay      = posSigma[8]
        errPub.pos_3_sigma_b_az      = posSigma[9]
        errPub.pos_3_sigma_b_rho_lm1 = posSigma[10]
        errPub.pos_3_sigma_b_rho_lm2 = posSigma[11]
        errPub.pos_3_sigma_b_rho_lm3 = posSigma[12]
        errPub.pos_3_sigma_b_rho_lm4 = posSigma[13]
        errPub.pos_3_sigma_b_rho_lm5 = posSigma[14]
        #errPub.pos_3_sigma_b_rho_lm6 = posSigma[15]
        errPub.pos_3_sigma_b_rho_v1  = posSigma[15]
        errPub.pos_3_sigma_b_rho_v2  = posSigma[16]

        errPub.neg_3_sigma_pn        = negSigma[0]
        errPub.neg_3_sigma_pe        = negSigma[1]
        errPub.neg_3_sigma_pd        = negSigma[2]
        errPub.neg_3_sigma_psi       = negSigma[3]
        errPub.neg_3_sigma_u         = negSigma[4]
        errPub.neg_3_sigma_v         = negSigma[5]
        errPub.neg_3_sigma_w         = negSigma[6]
        errPub.neg_3_sigma_b_ax      = negSigma[7]
        errPub.neg_3_sigma_b_ay      = negSigma[8]
        errPub.neg_3_sigma_b_az      = negSigma[9]
        errPub.neg_3_sigma_b_rho_lm1 = negSigma[10]
        errPub.neg_3_sigma_b_rho_lm2 = negSigma[11]
        errPub.neg_3_sigma_b_rho_lm3 = negSigma[12]
        errPub.neg_3_sigma_b_rho_lm4 = negSigma[13]
        errPub.neg_3_sigma_b_rho_lm5 = negSigma[14]
        #errPub.neg_3_sigma_b_rho_lm6 = negSigma[15]
        errPub.neg_3_sigma_b_rho_v1  = negSigma[15]
        errPub.neg_3_sigma_b_rho_v2  = negSigma[16]

        posPub = localization_estimates()
        posPub.header = gpsMsg.header
        posPub.header.stamp = gpsMsg.header.stamp
        posPub.pn = loc_pos[0]
        posPub.pe = loc_pos[1]
        posPub.pd = loc_pos[2] + val
        posPub.psi = (180.0/np.pi)*self.wrap_to_pi(yaw)
        posPub.u = odomMsg.twist.twist.linear.x
        posPub.v = odomMsg.twist.twist.linear.y
        posPub.w = odomMsg.twist.twist.linear.z

        return estPub, errPub, posPub

    def localize_UAV(self):
        while not rospy.is_shutdown():
            # Set coordinates for local origin
            if self.local_origin == None:
                self.setLocalOrigin()
                self.setLandmarkPosition()

            # Receive GPS lock and set local position
            if self.UAS1_FirstGPS == True:
                out = self.setFirstGPS(self.UAS1_GPSMsg)
                if out is not None:
                    self.UAS1_InitPos  = out
                    self.UAS1_FirstGPS = False
            if self.UAS2_FirstGPS == True:
                out = self.setFirstGPS(self.UAS2_GPSMsg)
                if out is not None:
                    self.UAS2_InitPos  = out
                    self.UAS2_FirstGPS = False
            if self.UAS3_FirstGPS == True:
                out = self.setFirstGPS(self.UAS3_GPSMsg)
                if out is not None:
                    self.UAS3_InitPos  = out
                    self.UAS3_FirstGPS = False
            
            # Estimator starts only after receiving GPS lock for all vehicles
            if (self.UAS1_FirstGPS == True or self.UAS2_FirstGPS == True or self.UAS3_FirstGPS == True):
                #rospy.loginfo("GPS lock not received. Node not initialized")
                self.rate.sleep()
                continue
            else:
                # Odometry measurements # used for comparison and initialization only
                if self.UAS1_OdomMsg is not None:
                    self.UAS1_u_true = self.UAS1_OdomMsg.twist.twist.linear.x
                    self.UAS1_v_true = self.UAS1_OdomMsg.twist.twist.linear.y
                    self.UAS1_w_true = self.UAS1_OdomMsg.twist.twist.linear.z
                # Update accelerometer measurements
                if self.UAS1_ImuMsg is not None:
                    # linear accelerations
                    self.UAS1_acc_u  = self.UAS1_ImuMsg.linear_acceleration.x
                    self.UAS1_acc_v  = self.UAS1_ImuMsg.linear_acceleration.y
                    self.UAS1_acc_w  = self.UAS1_ImuMsg.linear_acceleration.z - self.gravity
                    # angular velocity
                    self.UAS1_p      = self.UAS1_ImuMsg.angular_velocity.x
                    self.UAS1_q      = self.UAS1_ImuMsg.angular_velocity.y
                    self.UAS1_r      = self.UAS1_ImuMsg.angular_velocity.z
                    # orientation
                    quaternion         = self.UAS1_ImuMsg.orientation
                    self.UAS1_Roll, self.UAS1_Pitch, self.UAS1_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    self.UAS1_Time    = (self.UAS1_ImuMsg.header.stamp).to_sec()
                    if self.UAS1_FirstState == True:
                        id                    = 0
                        ns                    = self.NumStates
                        self.xHat[ns*id+0,0]  = self.UAS1_InitPos[0]
                        self.xHat[ns*id+1,0]  = self.UAS1_InitPos[1]
                        self.xHat[ns*id+2,0]  = self.UAS1_InitPos[2]
                        self.xHat[ns*id+3,0]  = (self.UAS1_Yaw)
                        self.xHat[ns*id+4,0]  = self.UAS1_u_true
                        self.xHat[ns*id+5,0]  = self.UAS1_v_true
                        self.xHat[ns*id+6,0]  = self.UAS1_w_true
                        self.UAS1_FirstState = False
                        self.currTime         = rospy.Time.now()
                        self.prevTime         = rospy.Time.now()
                # UAS2_ measurements
                if self.UAS2_OdomMsg is not None:
                    self.UAS2_u_true = self.UAS2_OdomMsg.twist.twist.linear.x
                    self.UAS2_v_true = self.UAS2_OdomMsg.twist.twist.linear.y
                    self.UAS2_w_true = self.UAS2_OdomMsg.twist.twist.linear.z
                # Update accelerometer measurements
                if self.UAS2_ImuMsg is not None:
                    # linear accelerations
                    self.UAS2_acc_u  = self.UAS2_ImuMsg.linear_acceleration.x
                    self.UAS2_acc_v  = self.UAS2_ImuMsg.linear_acceleration.y
                    self.UAS2_acc_w  = self.UAS2_ImuMsg.linear_acceleration.z - self.gravity
                    # angular velocity
                    self.UAS2_p      = self.UAS2_ImuMsg.angular_velocity.x
                    self.UAS2_q      = self.UAS2_ImuMsg.angular_velocity.y
                    self.UAS2_r      = self.UAS2_ImuMsg.angular_velocity.z
                    # orientation
                    quaternion          = self.UAS2_ImuMsg.orientation
                    self.UAS2_Roll, self.UAS2_Pitch, self.UAS2_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    self.UAS2_Time    = (self.UAS2_ImuMsg.header.stamp).to_sec()
                    if self.UAS2_FirstState == True:
                        id                     = 1
                        ns                     = self.NumStates
                        self.xHat[ns*id+0,0]   = self.UAS2_InitPos[0]
                        self.xHat[ns*id+1,0]   = self.UAS2_InitPos[1]
                        self.xHat[ns*id+2,0]   = self.UAS2_InitPos[2]
                        self.xHat[ns*id+3,0]   = (self.UAS2_Yaw)
                        self.xHat[ns*id+4,0]   = self.UAS2_u_true
                        self.xHat[ns*id+5,0]   = self.UAS2_v_true
                        self.xHat[ns*id+6,0]   = self.UAS2_w_true
                        self.UAS2_FirstState = False
                        self.currTime          = rospy.Time.now()
                        self.prevTime          = rospy.Time.now()
                # UAS3_ measurements
                if self.UAS3_OdomMsg is not None:
                    self.UAS3_u_true = self.UAS3_OdomMsg.twist.twist.linear.x
                    self.UAS3_v_true = self.UAS3_OdomMsg.twist.twist.linear.y
                    self.UAS3_w_true = self.UAS3_OdomMsg.twist.twist.linear.z
                # Update accelerometer measurements
                if self.UAS3_ImuMsg is not None:
                    # linear accelerations
                    self.UAS3_acc_u  = self.UAS3_ImuMsg.linear_acceleration.x
                    self.UAS3_acc_v  = self.UAS3_ImuMsg.linear_acceleration.y
                    self.UAS3_acc_w  = self.UAS3_ImuMsg.linear_acceleration.z - self.gravity
                    # angular velocity
                    self.UAS3_p      = self.UAS3_ImuMsg.angular_velocity.x
                    self.UAS3_q      = self.UAS3_ImuMsg.angular_velocity.y
                    self.UAS3_r      = self.UAS3_ImuMsg.angular_velocity.z
                    # orientation
                    quaternion          = self.UAS3_ImuMsg.orientation
                    self.UAS3_Roll, self.UAS3_Pitch, self.UAS3_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    self.UAS3_Time    = (self.UAS3_ImuMsg.header.stamp).to_sec()
                    if self.UAS3_FirstState == True:
                        id                     = 2
                        ns                     = self.NumStates
                        self.xHat[ns*id+0,0]   = self.UAS3_InitPos[0]
                        self.xHat[ns*id+1,0]   = self.UAS3_InitPos[1]
                        self.xHat[ns*id+2,0]   = self.UAS3_InitPos[2]
                        self.xHat[ns*id+3,0]   = (self.UAS3_Yaw)
                        self.xHat[ns*id+4,0]   = self.UAS3_u_true
                        self.xHat[ns*id+5,0]   = self.UAS3_v_true
                        self.xHat[ns*id+6,0]   = self.UAS3_w_true
                        self.UAS3_FirstState = False
                        self.currTime          = rospy.Time.now()
                        self.prevTime          = rospy.Time.now()

                if (self.UAS1_FirstState == False and self.UAS2_FirstState == False and self.UAS3_FirstState == False):
                    self.currTime = rospy.Time.now()
                    Ts = (self.currTime-self.prevTime).to_sec()
                    #print(Ts)
                    self.prevTime  = self.currTime
                    self.UAS1_Time = self.UAS1_Time + Ts
                    self.UAS2_Time = self.UAS2_Time + Ts
                    self.UAS3_Time = self.UAS3_Time + Ts
                    self.prediction(Ts)
                    # GPS measurement update and error calculations
                    if self.UAS1_GPSMsg is not None:
                        # Calculate error with respect to GPS measurements
                        UAS1_Err, UAS1_sigma, UAS1_pos_3sigma, UAS1_neg_3sigma, UAS1_local_pos = self.calculate_error(self.UAS1_GPSMsg,np.array([self.UAS1_u_true,self.UAS1_v_true,self.UAS1_w_true],dtype=np.float64),self.UAS1_Yaw,"UAS1")
                        if self.UAS1_GPS_flag is True:
                            self.gpsMeasurementUpdate('UAS1',self.UAS1_GPSMsg)
                        elif self.heightFlag is True:
                            self.heightMeasurementUpdate('UAS1',self.UAS1_GPSMsg)
                    if self.UAS2_GPSMsg is not None:
                        # Calculate error with respect to GPS measurements
                        UAS2_Err, UAS2_sigma, UAS2_pos_3sigma, UAS2_neg_3sigma, UAS2_local_pos = self.calculate_error(self.UAS2_GPSMsg,np.array([self.UAS2_u_true,self.UAS2_v_true,self.UAS2_w_true],dtype=np.float64),self.UAS2_Yaw,"UAS2")
                        if self.UAS2_GPS_flag is True:
                            self.gpsMeasurementUpdate('UAS2',self.UAS2_GPSMsg)
                        elif self.heightFlag is True:
                            self.heightMeasurementUpdate('UAS2',self.UAS2_GPSMsg)
                    if self.UAS3_GPSMsg is not None:
                        # Calculate error with respect to GPS measurements
                        UAS3_Err, UAS3_sigma, UAS3_pos_3sigma, UAS3_neg_3sigma, UAS3_local_pos = self.calculate_error(self.UAS3_GPSMsg,np.array([self.UAS3_u_true,self.UAS3_v_true,self.UAS3_w_true],dtype=np.float64),self.UAS3_Yaw,"UAS3")
                        if self.UAS3_GPS_flag is True:
                            self.gpsMeasurementUpdate('UAS3',self.UAS3_GPSMsg)
                        elif self.heightFlag is True:
                            self.heightMeasurementUpdate('UAS3',self.UAS3_GPSMsg)

                    # Landmark Measurement Updates
                    if self.UAS1_LM1_flag is True and self.lm1_to_UAS1_msg is not None:
                        if (self.lm1_to_UAS1_msg.range>2.5) and (self.lm1_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm1_to_UAS1_msg,self.lm1,'lm1')
                    if self.UAS1_LM2_flag is True and self.lm2_to_UAS1_msg is not None:
                        if (self.lm2_to_UAS1_msg.range>2.5) and (self.lm2_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm2_to_UAS1_msg,self.lm2,'lm2')
                    if self.UAS1_LM3_flag is True and self.lm3_to_UAS1_msg is not None:
                        if (self.lm3_to_UAS1_msg.range>2.5) and (self.lm3_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm3_to_UAS1_msg,self.lm3,'lm3')
                    if self.UAS1_LM4_flag is True and self.lm4_to_UAS1_msg is not None:
                        if (self.lm4_to_UAS1_msg.range>2.5) and (self.lm4_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm4_to_UAS1_msg,self.lm4,'lm4')
                    if self.UAS1_LM5_flag is True and self.lm5_to_UAS1_msg is not None:
                        if (self.lm5_to_UAS1_msg.range>2.5) and (self.lm5_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm5_to_UAS1_msg,self.lm5,'lm5')
                    if self.UAS1_LM6_flag is True and self.lm6_to_UAS1_msg is not None:
                        if (self.lm6_to_UAS1_msg.range>2.5) and (self.lm6_to_UAS1_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm6_to_UAS1_msg,self.lm6,'lm6')
                    
                    if self.UAS2_LM1_flag is True and self.lm1_to_UAS2_msg is not None:
                        if (self.lm1_to_UAS2_msg.range>2.5) and (self.lm1_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm1_to_UAS2_msg,self.lm1,'lm1')
                    if self.UAS2_LM2_flag is True and self.lm2_to_UAS2_msg is not None:
                        if (self.lm2_to_UAS2_msg.range>2.5) and (self.lm2_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm2_to_UAS2_msg,self.lm2,'lm2')
                    if self.UAS2_LM3_flag is True and self.lm3_to_UAS2_msg is not None:
                        if (self.lm3_to_UAS2_msg.range>2.5) and (self.lm3_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm3_to_UAS2_msg,self.lm3,'lm3')
                    if self.UAS2_LM4_flag is True and self.lm4_to_UAS2_msg is not None:
                        if (self.lm4_to_UAS2_msg.range>2.5) and (self.lm4_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm4_to_UAS2_msg,self.lm4,'lm4')
                    if self.UAS2_LM5_flag is True and self.lm5_to_UAS2_msg is not None:
                        if (self.lm5_to_UAS2_msg.range>2.5) and (self.lm5_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm5_to_UAS2_msg,self.lm5,'lm5')
                    if self.UAS2_LM6_flag is True and self.lm6_to_UAS2_msg is not None:
                        if (self.lm6_to_UAS2_msg.range>2.5) and (self.lm6_to_UAS2_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm6_to_UAS2_msg,self.lm6,'lm6')
            
                    if self.UAS3_LM1_flag is True and self.lm1_to_UAS3_msg is not None:
                        if (self.lm1_to_UAS3_msg.range>2.5) and (self.lm1_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm1_to_UAS3_msg,self.lm1,'lm1')
                    if self.UAS3_LM2_flag is True and self.lm2_to_UAS3_msg is not None:
                        if (self.lm2_to_UAS3_msg.range>2.5) and (self.lm2_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm2_to_UAS3_msg,self.lm2,'lm2')
                    if self.UAS3_LM3_flag is True and self.lm3_to_UAS3_msg is not None:
                        if (self.lm3_to_UAS3_msg.range>2.5) and (self.lm3_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm3_to_UAS3_msg,self.lm3,'lm3')
                    if self.UAS3_LM4_flag is True and self.lm4_to_UAS3_msg is not None:
                        if (self.lm4_to_UAS3_msg.range>2.5) and (self.lm4_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm4_to_UAS3_msg,self.lm4,'lm4')
                    if self.UAS3_LM5_flag is True and self.lm5_to_UAS3_msg is not None:
                        if (self.lm5_to_UAS3_msg.range>2.5) and (self.lm5_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm5_to_UAS3_msg,self.lm5,'lm5')
                    if self.UAS3_LM6_flag is True and self.lm6_to_UAS3_msg is not None:
                        if (self.lm6_to_UAS3_msg.range>2.5) and (self.lm6_to_UAS3_msg.range<self.rhoMax):
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm6_to_UAS3_msg,self.lm6,'lm6')

                    # Odometry
                    if self.odomFlag is True:
                        if self.UAS1_OdomMsg is not None:
                            self.velocityMeasurementUpdate('UAS1',self.UAS1_OdomMsg)
                        if self.UAS2_OdomMsg is not None:
                            self.velocityMeasurementUpdate('UAS2',self.UAS2_OdomMsg)
                        if self.UAS3_OdomMsg is not None:
                            self.velocityMeasurementUpdate('UAS3',self.UAS3_OdomMsg)
                    # Inter vehicle measurement updates
                    if self.UAS1_2_UAS2_Flag is True:
                        if self.UAS2_to_UAS1_msg is not None:
                            if self.UAS2_to_UAS1_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS1','UAS2',self.UAS2_to_UAS1_msg)
                        if self.UAS1_to_UAS2_msg is not None:
                            if self.UAS1_to_UAS2_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS2','UAS1',self.UAS1_to_UAS2_msg)
                    if self.UAS2_2_UAS3_Flag is True:
                        if self.UAS3_to_UAS2_msg is not None:
                            if self.UAS3_to_UAS2_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS2','UAS3',self.UAS3_to_UAS2_msg)
                        if self.UAS2_to_UAS3_msg is not None:
                            if self.UAS2_to_UAS3_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS3','UAS2',self.UAS2_to_UAS3_msg)
                    if self.UAS1_2_UAS3_Flag is True:
                        if self.UAS3_to_UAS1_msg is not None:
                            if self.UAS3_to_UAS1_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS1','UAS3',self.UAS3_to_UAS1_msg)
                        if self.UAS1_to_UAS3_msg is not None:
                            if self.UAS1_to_UAS3_msg.range < self.rhoMax:
                                self.intervehicleRangeMeasurementUpdate('UAS3','UAS1',self.UAS1_to_UAS3_msg)

                    # Publish
                    UAS1_EstPub, UAS1_ErrPub, UAS1_LocGPSPub = self.publish_msg('UAS1',UAS1_Err,UAS1_sigma,UAS1_pos_3sigma,UAS1_neg_3sigma,UAS1_local_pos)
                    self.UAS1_EstPub.publish(UAS1_EstPub)
                    self.UAS1_ErrPub.publish(UAS1_ErrPub)
                    self.UAS1_GPSPub.publish(UAS1_LocGPSPub)
                    UAS2_EstPub, UAS2_ErrPub, UAS2_LocGPSPub = self.publish_msg('UAS2',UAS2_Err,UAS2_sigma,UAS2_pos_3sigma,UAS2_neg_3sigma,UAS2_local_pos)
                    self.UAS2_EstPub.publish(UAS2_EstPub)
                    self.UAS2_ErrPub.publish(UAS2_ErrPub)
                    self.UAS2_GPSPub.publish(UAS2_LocGPSPub)
                    UAS3_EstPub, UAS3_ErrPub, UAS3_LocGPSPub = self.publish_msg('UAS3',UAS3_Err,UAS3_sigma,UAS3_pos_3sigma,UAS3_neg_3sigma,UAS3_local_pos)
                    self.UAS3_EstPub.publish(UAS3_EstPub)
                    self.UAS3_ErrPub.publish(UAS3_ErrPub)
                    self.UAS3_GPSPub.publish(UAS3_LocGPSPub)
            self.rate.sleep()

@dataclass
class Local_Coordinate:
	x: float
	y: float
	z: float
	zone: float
	letter: str

def main():
    qe = StateEstimator()
    try:
        qe.localize_UAV()
    except rospy.ROSInterruptException():
        rospy.loginfo("Cannot publish estimate. Shutting down node")
        return
    
if __name__=="__main__":
    main()