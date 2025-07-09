#!/usr/bin/env python3

# State Estimator with GPS for "UAS1_" and "UAS2_"
# Estimates position in North, East, Down and Heading
# External Sensor Measurements: GPS
# Internal Measurements: North, east, down velocities, phi, theta, p, q, r

import rospy
import numpy as np
from numpy import sin, cos, tan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Range
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Range
from mavros_msgs.msg import GPSRAW
from localization.msg import localization_estimates
from localization.msg import localization_error
import utm
from dataclasses import dataclass

class StateEstimator():
    def __init__(self):
        rospy.init_node("CoopLoc_range")
        rospy.loginfo("Range CL node initialized")

        # Initialize Variables
        self.UAS1_FirstState     = True
        self.UAS2_FirstState     = True
        self.UAS3_FirstState     = True
        self.UAS1_FirstGPS       = True
        self.UAS2_FirstGPS       = True
        self.UAS3_FirstGPS       = True
        self.NumStates           = 11
        self.NumVeh              = 3
        self.NumLM               = 5
        ns                       = self.NumStates
        nv                       = self.NumVeh
        self.xHat                = np.zeros((ns*nv,1), dtype=float)
        self.pCov                = 5.0*np.identity(ns*nv)
        self.pCov[ns*0+3,ns*0+3] = 0.1
        self.pCov[ns*1+3,ns*1+3] = 0.1
        self.pCov[ns*2+3,ns*2+3] = 0.1
        self.tPr                 = None
        self.local_origin        = None
        self.UAS1_Time           = None
        self.UAS2_Time           = None
        self.UAS3_Time           = None

        # Measurement Flags
        self.UAS1_GPSflag        = False
        self.UAS2_GPSflag        = False
        self.UAS3_GPSflag        = False
        self.heightFlag          = True

        self.rangeflag           = True
        self.coopflag            = True

        # Noise Parameters
        self.sigmaVelX_UAS1     = 0.005
        self.sigmaVelY_UAS1     = 0.005
        self.sigmaVelZ_UAS1     = 0.5
        self.sigmaRoll_UAS1     = 0.002
        self.sigmaPitch_UAS1    = 0.002
        self.sigmaOmg_UAS1      = 0.1
        self.sigmaVelX_UAS2     = 0.005
        self.sigmaVelY_UAS2     = 0.005
        self.sigmaVelZ_UAS2     = 0.5
        self.sigmaRoll_UAS2     = 0.002
        self.sigmaPitch_UAS2    = 0.002
        self.sigmaOmg_UAS2      = 0.1
        self.sigmaVelX_UAS3     = 0.005
        self.sigmaVelY_UAS3     = 0.005
        self.sigmaVelZ_UAS3     = 0.5
        self.sigmaRoll_UAS3     = 0.002
        self.sigmaPitch_UAS3    = 0.002
        self.sigmaOmg_UAS3      = 0.1
        self.gpsSigma           = 2.0

        #
        self.rhoTimeDiff         = 0.1
        self.rhoSigmaLM          = 3.0
        self.rhoSigmav2v         = 4.0

        # Subscribers
        self.UAS1_OdomMsg         = None
        self.UAS1_GPSMsg          = None
        self.UAS2_OdomMsg         = None
        self.UAS2_GPSMsg          = None
        self.UAS3_OdomMsg         = None
        self.UAS3_GPSMsg          = None

        self.lm1_to_UAS1_msg      = None
        self.lm2_to_UAS1_msg      = None
        self.lm3_to_UAS1_msg      = None
        self.lm4_to_UAS1_msg      = None
        self.lm5_to_UAS1_msg      = None
        self.lm1_to_UAS2_msg      = None
        self.lm2_to_UAS2_msg      = None
        self.lm3_to_UAS2_msg      = None
        self.lm4_to_UAS2_msg      = None
        self.lm5_to_UAS2_msg      = None
        self.lm1_to_UAS3_msg      = None
        self.lm2_to_UAS3_msg      = None
        self.lm3_to_UAS3_msg      = None
        self.lm4_to_UAS3_msg      = None
        self.lm5_to_UAS3_msg      = None
        self.UAS3_to_UAS1_msg     = None
        self.UAS2_to_UAS1_msg     = None
        self.UAS1_to_UAS2_msg     = None
        self.UAS3_to_UAS2_msg     = None
        self.UAS1_to_UAS3_msg     = None
        self.UAS2_to_UAS3_msg     = None


        self.UAS1_OdomSub         = rospy.Subscriber('/UAS1/odom', Odometry, self.UAS1_OdomCallback)
        self.UAS1_GPSSub          = rospy.Subscriber('/UAS1/GPS', NavSatFix, self.UAS1_GPSCallback)
        self.UAS2_OdomSub         = rospy.Subscriber('/UAS2/odom', Odometry, self.UAS2_OdomCallback)
        self.UAS2_GPSSub          = rospy.Subscriber('/UAS2/GPS', NavSatFix, self.UAS2_GPSCallback)
        self.UAS3_OdomSub         = rospy.Subscriber('/UAS3/odom', Odometry, self.UAS3_OdomCallback)
        self.UAS3_GPSSub          = rospy.Subscriber('/UAS3/GPS', NavSatFix, self.UAS3_GPSCallback)

        self.lm1_to_UAS1_sub      = rospy.Subscriber('/range_lm1_to_UAS1', Range, self.lm1_to_UAS1_callback)
        self.lm2_to_UAS1_sub      = rospy.Subscriber('/range_lm2_to_UAS1', Range, self.lm2_to_UAS1_callback)
        self.lm3_to_UAS1_sub      = rospy.Subscriber('/range_lm3_to_UAS1', Range, self.lm3_to_UAS1_callback)
        self.lm4_to_UAS1_sub      = rospy.Subscriber('/range_lm4_to_UAS1', Range, self.lm4_to_UAS1_callback)
        self.lm5_to_UAS1_sub      = rospy.Subscriber('/range_lm5_to_UAS1', Range, self.lm5_to_UAS1_callback)
        self.lm1_to_UAS2_sub      = rospy.Subscriber('/range_lm1_to_UAS2', Range, self.lm1_to_UAS2_callback)
        self.lm2_to_UAS2_sub      = rospy.Subscriber('/range_lm2_to_UAS2', Range, self.lm2_to_UAS2_callback)
        self.lm3_to_UAS2_sub      = rospy.Subscriber('/range_lm3_to_UAS2', Range, self.lm3_to_UAS2_callback)
        self.lm4_to_UAS2_sub      = rospy.Subscriber('/range_lm4_to_UAS2', Range, self.lm4_to_UAS2_callback)
        self.lm5_to_UAS2_sub      = rospy.Subscriber('/range_lm5_to_UAS2', Range, self.lm5_to_UAS2_callback)
        self.lm1_to_UAS3_sub      = rospy.Subscriber('/range_lm1_to_UAS3', Range, self.lm1_to_UAS3_callback)
        self.lm2_to_UAS3_sub      = rospy.Subscriber('/range_lm2_to_UAS3', Range, self.lm2_to_UAS3_callback)
        self.lm3_to_UAS3_sub      = rospy.Subscriber('/range_lm3_to_UAS3', Range, self.lm3_to_UAS3_callback)
        self.lm4_to_UAS3_sub      = rospy.Subscriber('/range_lm4_to_UAS3', Range, self.lm4_to_UAS3_callback)
        self.lm5_to_UAS3_sub      = rospy.Subscriber('/range_lm5_to_UAS3', Range, self.lm5_to_UAS3_callback)

        self.UAS2_to_UAS1_sub  = rospy.Subscriber('/range_UAS2_to_UAS1', Range, self.UAS2_to_UAS1_callback)
        self.UAS3_to_UAS1_sub  = rospy.Subscriber('/range_UAS3_to_UAS1', Range, self.UAS3_to_UAS1_callback)
        self.UAS1_to_UAS2_sub  = rospy.Subscriber('/range_UAS1_to_UAS2', Range, self.UAS1_to_UAS2_callback)
        self.UAS3_to_UAS2_sub  = rospy.Subscriber('/range_UAS3_to_UAS2', Range, self.UAS3_to_UAS2_callback)
        self.UAS1_to_UAS2_sub  = rospy.Subscriber('/range_UAS1_to_UAS3', Range, self.UAS1_to_UAS3_callback)
        self.UAS3_to_UAS2_sub  = rospy.Subscriber('/range_UAS2_to_UAS3', Range, self.UAS2_to_UAS3_callback)

        # Publishers
        self.UAS1_EstPub = rospy.Publisher('/UAS1_estimated_states', localization_estimates, queue_size=1)
        self.UAS1_ErrPub = rospy.Publisher('/UAS1_error_states', localization_error, queue_size=1)
        self.UAS1_GPSPub = rospy.Publisher('/UAS1_true_states', localization_estimates, queue_size=1)
        self.UAS2_EstPub = rospy.Publisher('/UAS2_estimated_states', localization_estimates, queue_size=1)
        self.UAS2_ErrPub = rospy.Publisher('/UAS2_error_states', localization_error, queue_size=1)
        self.UAS2_GPSPub = rospy.Publisher('/UAS2_true_states', localization_estimates, queue_size=1)
        self.UAS3_EstPub = rospy.Publisher('/UAS3_estimated_states', localization_estimates, queue_size=1)
        self.UAS3_ErrPub = rospy.Publisher('/UAS3_error_states', localization_error, queue_size=1)
        self.UAS3_GPSPub = rospy.Publisher('/UAS3_true_states', localization_estimates, queue_size=1)

        # rate
        self.rate         = rospy.Rate(20)
        self.steps        = 10

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
        self.lm1 = np.array([lm1_x, lm1_y, lm1_z], dtype=float)
        rospy.loginfo("Landmark 1 position is set")
        print("Landmark 1:",self.lm1)

        lm2_lat = 39.1547440
        lm2_lon = -84.7880159
        lm2_alt = 166.700 + 1.605
        lm2_pos  = utm.from_latlon(lm2_lat,lm2_lon)
        lm2_x, lm2_y, lm2_z = self.convert_to_local(lm2_pos[0], lm2_pos[1], lm2_alt)
        self.lm2 = np.array([lm2_x, lm2_y, lm2_z], dtype=float)
        rospy.loginfo("Landmark 2 position is set")
        print("Landmark 2:",self.lm2)

        lm3_lat = 39.1546233
        lm3_lon = -84.7885084
        lm3_alt = 165.020 + 1.79
        lm3_pos  = utm.from_latlon(lm3_lat,lm3_lon)
        lm3_x, lm3_y, lm3_z = self.convert_to_local(lm3_pos[0], lm3_pos[1], lm3_alt)
        self.lm3 = np.array([lm3_x, lm3_y, lm3_z], dtype=float)
        rospy.loginfo("Landmark 3 position is set")
        print("Landmark 3:",self.lm3)

        lm4_lat = 39.1544874
        lm4_lon = -84.7888732
        lm4_alt = 164.540 + 1.75
        lm4_pos  = utm.from_latlon(lm4_lat,lm4_lon)
        lm4_x, lm4_y, lm4_z = self.convert_to_local(lm4_pos[0], lm4_pos[1], lm4_alt)
        self.lm4 = np.array([lm4_x, lm4_y, lm4_z], dtype=float)
        rospy.loginfo("Landmark 4 position is set")
        print("Landmark 4:",self.lm4)

        lm5_lat = 39.1542826
        lm5_lon = -84.7885955
        lm5_alt = 165.230 + 1.62
        lm5_pos  = utm.from_latlon(lm5_lat,lm5_lon)
        lm5_x, lm5_y, lm5_z = self.convert_to_local(lm5_pos[0], lm5_pos[1], lm5_alt)
        self.lm5 = np.array([lm5_x, lm5_y, lm5_z], dtype=float)
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
            return np.array([xPos,yPos,zPos],dtype=float)
        else:
            return None
        
    def wrap_to_pi(self,ang):
        return (ang+np.pi) % (2*np.pi) - np.pi

    def UAS1_GPSCallback(self,msg):
        self.UAS1_GPSMsg = msg

    def UAS1_OdomCallback(self,msg):
        self.UAS1_OdomMsg = msg

    def UAS2_GPSCallback(self,msg):
        self.UAS2_GPSMsg = msg

    def UAS2_OdomCallback(self,msg):
        self.UAS2_OdomMsg = msg

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
        ns = self.NumStates
        nv = self.NumVeh
        for i in range(self.steps):
            id             = 0
            UAS1_Psi       = self.xHat[ns*id+3,0]
            UAS1_Phi       = self.UAS1_Roll
            UAS1_Theta     = self.UAS1_Pitch
            UAS1_R_mat     = np.array([[cos(UAS1_Theta)*cos(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Theta)*cos(UAS1_Psi)-cos(UAS1_Phi)*sin(UAS1_Psi), cos(UAS1_Phi)*sin(UAS1_Theta)*cos(UAS1_Psi)+sin(UAS1_Phi)*sin(UAS1_Psi)],
                                       [cos(UAS1_Theta)*sin(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Theta)*sin(UAS1_Psi)+cos(UAS1_Phi)*cos(UAS1_Psi), cos(UAS1_Phi)*sin(UAS1_Theta)*sin(UAS1_Psi)-sin(UAS1_Phi)*cos(UAS1_Psi)],
                                       [             -sin(UAS1_Theta),                                           sin(UAS1_Phi)*cos(UAS1_Theta),                                           cos(UAS1_Phi)*cos(UAS1_Theta)]],dtype=float)
            id             = 1
            UAS2_Psi       = self.xHat[ns*id+3,0]
            UAS2_Phi       = self.UAS2_Roll
            UAS2_Theta     = self.UAS2_Pitch
            UAS2_R_mat     = np.array([[cos(UAS2_Theta)*cos(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Theta)*cos(UAS2_Psi)-cos(UAS2_Phi)*sin(UAS2_Psi), cos(UAS2_Phi)*sin(UAS2_Theta)*cos(UAS2_Psi)+sin(UAS2_Phi)*sin(UAS2_Psi)],
                                       [cos(UAS2_Theta)*sin(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Theta)*sin(UAS2_Psi)+cos(UAS2_Phi)*cos(UAS2_Psi), cos(UAS2_Phi)*sin(UAS2_Theta)*sin(UAS2_Psi)-sin(UAS2_Phi)*cos(UAS2_Psi)],
                                       [             -sin(UAS2_Theta),                                           sin(UAS2_Phi)*cos(UAS2_Theta),                                           cos(UAS2_Phi)*cos(UAS2_Theta)]],dtype=float)
            id             = 2
            UAS3_Psi       = self.xHat[ns*id+3,0]
            UAS3_Phi       = self.UAS3_Roll
            UAS3_Theta     = self.UAS3_Pitch
            UAS3_R_mat     = np.array([[cos(UAS3_Theta)*cos(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Theta)*cos(UAS3_Psi)-cos(UAS3_Phi)*sin(UAS3_Psi), cos(UAS3_Phi)*sin(UAS3_Theta)*cos(UAS3_Psi)+sin(UAS3_Phi)*sin(UAS3_Psi)],
                                       [cos(UAS3_Theta)*sin(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Theta)*sin(UAS3_Psi)+cos(UAS3_Phi)*cos(UAS3_Psi), cos(UAS3_Phi)*sin(UAS3_Theta)*sin(UAS3_Psi)-sin(UAS3_Phi)*cos(UAS3_Psi)],
                                       [             -sin(UAS3_Theta),                                           sin(UAS3_Phi)*cos(UAS3_Theta),                                           cos(UAS3_Phi)*cos(UAS3_Theta)]],dtype=float)

            UAS1_pos_dot   = np.matmul(UAS1_R_mat,np.array([[self.UAS1_VelX],[self.UAS1_VelY],[self.UAS1_VelZ]],dtype=float))
            UAS1_psi_dot   = np.array([[sin(UAS1_Phi)*(1/cos(UAS1_Theta))*self.UAS1_Q + cos(UAS1_Phi)*(1/cos(UAS1_Theta))*self.UAS1_R]],dtype=float)
            UAS1_bias_dot  = np.zeros((7,1),dtype=float)
            UAS2_pos_dot   = np.matmul(UAS2_R_mat,np.array([[self.UAS2_VelX],[self.UAS2_VelY],[self.UAS2_VelZ]],dtype=float))
            UAS2_psi_dot   = np.array([[sin(UAS2_Phi)*(1/cos(UAS2_Theta))*self.UAS2_Q + cos(UAS2_Phi)*(1/cos(UAS2_Theta))*self.UAS2_R]],dtype=float)
            UAS2_bias_dot  = np.zeros((7,1),dtype=float)
            UAS3_pos_dot   = np.matmul(UAS3_R_mat,np.array([[self.UAS3_VelX],[self.UAS3_VelY],[self.UAS3_VelZ]],dtype=float))
            UAS3_psi_dot   = np.array([[sin(UAS3_Phi)*(1/cos(UAS3_Theta))*self.UAS3_Q + cos(UAS3_Phi)*(1/cos(UAS3_Theta))*self.UAS3_R]],dtype=float)
            UAS3_bias_dot  = np.zeros((7,1),dtype=float)

            self.xHat      = self.xHat + (Ts/self.steps)*np.concatenate((UAS1_pos_dot,UAS1_psi_dot,UAS1_bias_dot,
                                                                         UAS2_pos_dot,UAS2_psi_dot,UAS2_bias_dot,
                                                                         UAS3_pos_dot,UAS3_psi_dot,UAS3_bias_dot),axis=0)

            At             = np.zeros((ns*nv,ns*nv),dtype=float)
            UAS1_At        = np.array([[0, 0, 0, self.UAS1_VelZ*(cos(UAS1_Psi)*sin(UAS1_Phi) - cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - self.UAS1_VelY*(cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - self.UAS1_VelX*cos(UAS1_Theta)*sin(UAS1_Psi), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, self.UAS1_VelZ*(sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta)) - self.UAS1_VelY*(cos(UAS1_Phi)*sin(UAS1_Psi) - cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta)) + self.UAS1_VelX*cos(UAS1_Psi)*cos(UAS1_Theta), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
              
            UAS2_At        = np.array([[0, 0, 0, self.UAS2_VelZ*(cos(UAS2_Psi)*sin(UAS2_Phi) - cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - self.UAS2_VelY*(cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - self.UAS2_VelX*cos(UAS2_Theta)*sin(UAS2_Psi), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, self.UAS2_VelZ*(sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta)) - self.UAS2_VelY*(cos(UAS2_Phi)*sin(UAS2_Psi) - cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta)) + self.UAS2_VelX*cos(UAS2_Psi)*cos(UAS2_Theta), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
                     
            UAS3_At        = np.array([[0, 0, 0, self.UAS3_VelZ*(cos(UAS3_Psi)*sin(UAS3_Phi) - cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - self.UAS3_VelY*(cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - self.UAS3_VelX*cos(UAS3_Theta)*sin(UAS3_Psi), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, self.UAS3_VelZ*(sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta)) - self.UAS3_VelY*(cos(UAS3_Phi)*sin(UAS3_Psi) - cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta)) + self.UAS3_VelX*cos(UAS3_Psi)*cos(UAS3_Theta), 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,                                                                                                                                                                                                                                      0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

            id                                       = 0                    
            At[ns*id+0:ns*id+ns,ns*id+0:ns*id+ns]    = UAS1_At
            id                                       = 1
            At[ns*id+0:ns*id+ns,ns*id+0:ns*id+ns]    = UAS2_At
            id                                       = 2
            At[ns*id+0:ns*id+ns,ns*id+0:ns*id+ns]    = UAS3_At

            Bt            = np.zeros((ns*nv,24),dtype=float)
                
            UAS1_Bt       = np.array([[cos(UAS1_Psi)*cos(UAS1_Theta), cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta) - cos(UAS1_Phi)*sin(UAS1_Psi), sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta),   self.UAS1_VelY*(sin(UAS1_Phi)*sin(UAS1_Psi) + cos(UAS1_Phi)*cos(UAS1_Psi)*sin(UAS1_Theta)) + self.UAS1_VelZ*(cos(UAS1_Phi)*sin(UAS1_Psi) - cos(UAS1_Psi)*sin(UAS1_Phi)*sin(UAS1_Theta)), cos(UAS1_Psi)*(self.UAS1_VelZ*cos(UAS1_Phi)*cos(UAS1_Theta) - self.UAS1_VelX*sin(UAS1_Theta) + self.UAS1_VelY*cos(UAS1_Theta)*sin(UAS1_Phi)), 0,                             0,                             0],
                                      [cos(UAS1_Theta)*sin(UAS1_Psi), cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta), cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta) - cos(UAS1_Psi)*sin(UAS1_Phi), - self.UAS1_VelY*(cos(UAS1_Psi)*sin(UAS1_Phi) - cos(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)) - self.UAS1_VelZ*(cos(UAS1_Phi)*cos(UAS1_Psi) + sin(UAS1_Phi)*sin(UAS1_Psi)*sin(UAS1_Theta)), sin(UAS1_Psi)*(self.UAS1_VelZ*cos(UAS1_Phi)*cos(UAS1_Theta) - self.UAS1_VelX*sin(UAS1_Theta) + self.UAS1_VelY*cos(UAS1_Theta)*sin(UAS1_Phi)), 0,                             0,                             0],
                                      [             -sin(UAS1_Theta),                                             cos(UAS1_Theta)*sin(UAS1_Phi),                                             cos(UAS1_Phi)*cos(UAS1_Theta),                                                                                                             cos(UAS1_Theta)*(self.UAS1_VelY*cos(UAS1_Phi) - self.UAS1_VelZ*sin(UAS1_Phi)),               - self.UAS1_VelX*cos(UAS1_Theta) - self.UAS1_VelZ*cos(UAS1_Phi)*sin(UAS1_Theta) - self.UAS1_VelY*sin(UAS1_Phi)*sin(UAS1_Theta), 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                   (self.UAS1_Q*cos(UAS1_Phi) - self.UAS1_R*sin(UAS1_Phi))/cos(UAS1_Theta),                                               (sin(UAS1_Theta)*(self.UAS1_R*cos(UAS1_Phi) + self.UAS1_Q*sin(UAS1_Phi)))/(cos(UAS1_Theta))**2, 0, sin(UAS1_Phi)/cos(UAS1_Theta), cos(UAS1_Phi)/cos(UAS1_Theta)],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0]], dtype=float)
                        
            UAS2_Bt       = np.array([[cos(UAS2_Psi)*cos(UAS2_Theta), cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta) - cos(UAS2_Phi)*sin(UAS2_Psi), sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta),   self.UAS2_VelY*(sin(UAS2_Phi)*sin(UAS2_Psi) + cos(UAS2_Phi)*cos(UAS2_Psi)*sin(UAS2_Theta)) + self.UAS2_VelZ*(cos(UAS2_Phi)*sin(UAS2_Psi) - cos(UAS2_Psi)*sin(UAS2_Phi)*sin(UAS2_Theta)), cos(UAS2_Psi)*(self.UAS2_VelZ*cos(UAS2_Phi)*cos(UAS2_Theta) - self.UAS2_VelX*sin(UAS2_Theta) + self.UAS2_VelY*cos(UAS2_Theta)*sin(UAS2_Phi)), 0,                             0,                             0],
                                      [cos(UAS2_Theta)*sin(UAS2_Psi), cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta), cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta) - cos(UAS2_Psi)*sin(UAS2_Phi), - self.UAS2_VelY*(cos(UAS2_Psi)*sin(UAS2_Phi) - cos(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)) - self.UAS2_VelZ*(cos(UAS2_Phi)*cos(UAS2_Psi) + sin(UAS2_Phi)*sin(UAS2_Psi)*sin(UAS2_Theta)), sin(UAS2_Psi)*(self.UAS2_VelZ*cos(UAS2_Phi)*cos(UAS2_Theta) - self.UAS2_VelX*sin(UAS2_Theta) + self.UAS2_VelY*cos(UAS2_Theta)*sin(UAS2_Phi)), 0,                             0,                             0],
                                      [             -sin(UAS2_Theta),                                             cos(UAS2_Theta)*sin(UAS2_Phi),                                             cos(UAS2_Phi)*cos(UAS2_Theta),                                                                                                             cos(UAS2_Theta)*(self.UAS2_VelY*cos(UAS2_Phi) - self.UAS2_VelZ*sin(UAS2_Phi)),               - self.UAS2_VelX*cos(UAS2_Theta) - self.UAS2_VelZ*cos(UAS2_Phi)*sin(UAS2_Theta) - self.UAS2_VelY*sin(UAS2_Phi)*sin(UAS2_Theta), 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                   (self.UAS2_Q*cos(UAS2_Phi) - self.UAS2_R*sin(UAS2_Phi))/cos(UAS2_Theta),                                               (sin(UAS2_Theta)*(self.UAS2_R*cos(UAS2_Phi) + self.UAS2_Q*sin(UAS2_Phi)))/(cos(UAS2_Theta))**2, 0, sin(UAS2_Phi)/cos(UAS2_Theta), cos(UAS2_Phi)/cos(UAS2_Theta)],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0]], dtype=float)
                        
            UAS3_Bt       = np.array([[cos(UAS3_Psi)*cos(UAS3_Theta), cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta) - cos(UAS3_Phi)*sin(UAS3_Psi), sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta),   self.UAS3_VelY*(sin(UAS3_Phi)*sin(UAS3_Psi) + cos(UAS3_Phi)*cos(UAS3_Psi)*sin(UAS3_Theta)) + self.UAS3_VelZ*(cos(UAS3_Phi)*sin(UAS3_Psi) - cos(UAS3_Psi)*sin(UAS3_Phi)*sin(UAS3_Theta)), cos(UAS3_Psi)*(self.UAS3_VelZ*cos(UAS3_Phi)*cos(UAS3_Theta) - self.UAS3_VelX*sin(UAS3_Theta) + self.UAS3_VelY*cos(UAS3_Theta)*sin(UAS3_Phi)), 0,                             0,                             0],
                                      [cos(UAS3_Theta)*sin(UAS3_Psi), cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta), cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta) - cos(UAS3_Psi)*sin(UAS3_Phi), - self.UAS3_VelY*(cos(UAS3_Psi)*sin(UAS3_Phi) - cos(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)) - self.UAS3_VelZ*(cos(UAS3_Phi)*cos(UAS3_Psi) + sin(UAS3_Phi)*sin(UAS3_Psi)*sin(UAS3_Theta)), sin(UAS3_Psi)*(self.UAS3_VelZ*cos(UAS3_Phi)*cos(UAS3_Theta) - self.UAS3_VelX*sin(UAS3_Theta) + self.UAS3_VelY*cos(UAS3_Theta)*sin(UAS3_Phi)), 0,                             0,                             0],
                                      [             -sin(UAS3_Theta),                                             cos(UAS3_Theta)*sin(UAS3_Phi),                                             cos(UAS3_Phi)*cos(UAS3_Theta),                                                                                                             cos(UAS3_Theta)*(self.UAS3_VelY*cos(UAS3_Phi) - self.UAS3_VelZ*sin(UAS3_Phi)),               - self.UAS3_VelX*cos(UAS3_Theta) - self.UAS3_VelZ*cos(UAS3_Phi)*sin(UAS3_Theta) - self.UAS3_VelY*sin(UAS3_Phi)*sin(UAS3_Theta), 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                   (self.UAS3_Q*cos(UAS3_Phi) - self.UAS3_R*sin(UAS3_Phi))/cos(UAS3_Theta),                                               (sin(UAS3_Theta)*(self.UAS3_R*cos(UAS3_Phi) + self.UAS3_Q*sin(UAS3_Phi)))/(cos(UAS3_Theta))**2, 0, sin(UAS3_Phi)/cos(UAS3_Theta), cos(UAS3_Phi)/cos(UAS3_Theta)],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0],
                                      [                            0,                                                                         0,                                                                         0,                                                                                                                                                                                         0,                                                                                                                                            0, 0,                             0,                             0]], dtype=float)
            id                         = 0        
            Bt[ns*id+0:ns*id+ns,0:8]   = UAS1_Bt
            id                         = 1
            Bt[ns*id+0:ns*id+ns,8:16]  = UAS2_Bt
            id                         = 2
            Bt[ns*id+0:ns*id+ns,16:24] = UAS3_Bt

            UAS1_qu       = np.array([[self.sigmaVelX_UAS1**2, 0, 0, 0, 0, 0, 0, 0],
                                        [0, self.sigmaVelY_UAS1**2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, self.sigmaVelZ_UAS1**2, 0, 0, 0, 0, 0],
                                        [0, 0, 0, self.sigmaRoll_UAS1**2, 0, 0, 0, 0],
                                        [0, 0, 0, 0, self.sigmaPitch_UAS1**2, 0, 0, 0],
                                        [0, 0, 0, 0, 0, self.sigmaOmg_UAS1**2, 0, 0],
                                        [0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS1**2, 0],
                                        [0, 0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS1**2]], dtype=float)
            UAS2_qu       = np.array([[self.sigmaVelX_UAS2**2, 0, 0, 0, 0, 0, 0, 0],
                                        [0, self.sigmaVelY_UAS2**2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, self.sigmaVelZ_UAS2**2, 0, 0, 0, 0, 0],
                                        [0, 0, 0, self.sigmaRoll_UAS2**2, 0, 0, 0, 0],
                                        [0, 0, 0, 0, self.sigmaPitch_UAS2**2, 0, 0, 0],
                                        [0, 0, 0, 0, 0, self.sigmaOmg_UAS2**2, 0, 0],
                                        [0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS2**2, 0],
                                        [0, 0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS2**2]], dtype=float)
            UAS3_qu       = np.array([[self.sigmaVelX_UAS3**2, 0, 0, 0, 0, 0, 0, 0],
                                        [0, self.sigmaVelY_UAS3**2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, self.sigmaVelZ_UAS3**2, 0, 0, 0, 0, 0],
                                        [0, 0, 0, self.sigmaRoll_UAS3**2, 0, 0, 0, 0],
                                        [0, 0, 0, 0, self.sigmaPitch_UAS3**2, 0, 0, 0],
                                        [0, 0, 0, 0, 0, self.sigmaOmg_UAS3**2, 0, 0],
                                        [0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS3**2, 0],
                                        [0, 0, 0, 0, 0, 0, 0, self.sigmaOmg_UAS3**2]], dtype=float)
            QU              = np.zeros((24,24),dtype=float)
            QU[0:8,0:8]     = UAS1_qu
            QU[8:16,8:16]   = UAS2_qu
            QU[16:24,16:24] = UAS3_qu

            Qt              = np.matmul(Bt,np.matmul(QU,np.transpose(Bt)))
            id = 0
            Qt[ns*id+4,ns*id+4] = 0.4
            Qt[ns*id+5,ns*id+5] = 0.4
            Qt[ns*id+6,ns*id+6] = 0.4
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.8
            Qt[ns*id+10,ns*id+10] = 0.8
            id = 1
            Qt[ns*id+4,ns*id+4] = 0.4
            Qt[ns*id+5,ns*id+5] = 0.4
            Qt[ns*id+6,ns*id+6] = 0.4
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.8
            Qt[ns*id+10,ns*id+10] = 0.8
            id = 2
            Qt[ns*id+4,ns*id+4] = 0.4
            Qt[ns*id+5,ns*id+5] = 0.4
            Qt[ns*id+6,ns*id+6] = 0.4
            Qt[ns*id+7,ns*id+7] = 0.4
            Qt[ns*id+8,ns*id+8] = 0.4
            Qt[ns*id+9,ns*id+9] = 0.8
            Qt[ns*id+10,ns*id+10] = 0.8
            self.pCov       = self.pCov + (Ts/self.steps)*(np.matmul(At,self.pCov) + np.matmul(self.pCov,np.transpose(At)) + Qt)

    def gpsMeasurementUpdate(self,veh,msg):
        gpsTime = (msg.header.stamp).to_sec()
        ns      = self.NumStates
        nv      = self.NumVeh
        if veh == 'UAS1':
            odomTime = self.UAS1_Time
            id       = 0
            val      = -0
        if veh == 'UAS2':
            odomTime = self.UAS2_Time
            id       = 1
            val      = -0.0
        if veh == 'UAS3':
            odomTime = self.UAS3_Time
            val      = +0.0
            id       = 2
        
        timeDiff = abs(gpsTime-odomTime)
        if timeDiff < 0.1:
            lat              = float(msg.latitude)
            lon              = float(msg.longitude)
            alt              = float(msg.altitude)
            gpsPos           = utm.from_latlon(lat,lon)
            xPos, yPos, zPos = self.convert_to_local(gpsPos[0],gpsPos[1],alt)
            zt               = np.array([[xPos],[yPos],[zPos+val]],dtype=float)
            Ht               = np.zeros((3,ns*nv),dtype=float)
            Ht[0][ns*id+0]    = 1.0
            Ht[1][ns*id+1]    = 1.0
            Ht[2][ns*id+2]    = 1.0
            Rt               = np.array([[self.gpsSigma**2, 0, 0],
                                         [0, self.gpsSigma**2, 0],
                                         [0, 0, self.gpsSigma**2]],dtype=float)
            mat              = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
            inv_mat          = np.linalg.inv(mat)
            Lt               = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
            self.pCov        = np.dot(np.identity(ns*nv)-np.dot(Lt,Ht),self.pCov)
            self.xHat        = np.add(self.xHat,np.dot(Lt,zt-np.dot(Ht,self.xHat)))

    def heightMeasurementUpdate(self,veh,msg):
        gpsTime = (msg.header.stamp).to_sec()
        ns      = self.NumStates
        nv      = self.NumVeh
        if veh == 'UAS1':
            odomTime = self.UAS1_Time
            id       = 0
            val      = 0.0
        if veh == 'UAS2':
            odomTime = self.UAS2_Time
            id       = 1
            val      = -0.0
        if veh == 'UAS3':
            odomTime = self.UAS3_Time
            val      = +0.0
            id       = 2
        
        timeDiff = abs(gpsTime-odomTime)
        if timeDiff < 0.1:
            lat              = float(msg.latitude)
            lon              = float(msg.longitude)
            alt              = float(msg.altitude)
            gpsPos           = utm.from_latlon(lat,lon)
            xPos, yPos, zPos = self.convert_to_local(gpsPos[0],gpsPos[1],alt)
            zt               = np.array([[zPos+val]],dtype=float)
            Ht               = np.zeros((1,ns*nv),dtype=float)
            Ht[0][ns*id+2]   = 1.0
            Rt               = np.array([[self.gpsSigma**2]],dtype=float)
            mat              = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
            inv_mat          = np.linalg.inv(mat)
            Lt               = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
            self.pCov        = np.dot(np.identity(ns*nv)-np.dot(Lt,Ht),self.pCov)
            self.xHat        = np.add(self.xHat,np.dot(Lt,zt-np.dot(Ht,self.xHat)))

    def landmarkRangeMeasurementUpdate(self,veh,msg,lm,lmID):
        ns = self.NumStates
        nv = self.NumVeh
        if msg.range > 2.5:
            lmX      = lm[0]
            lmY      = lm[1]
            lmZ      = lm[2]
            Ht       = np.zeros((1,ns*nv),dtype=float)
            zt       = np.array([[msg.range/1000.0]],dtype=float)
            
            rhoTime = (msg.header.stamp).to_sec()

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
                bid = 4
                hid = 0
            if lmID == 'lm2':
                bid = 5
                hid = 1
            if lmID == 'lm3':
                bid = 6
                hid = 2
            if lmID == 'lm4':
                bid = 7
                hid = 3
            if lmID == 'lm5':
                bid = 8
                hid = 4

            xPos = self.xHat[ns*id+0,0]
            yPos = self.xHat[ns*id+1,0]
            zPos = self.xHat[ns*id+2,0]
            bias = self.xHat[ns*id+bid,0]

            xDiff = xPos-lmX
            yDiff = yPos-lmY
            zDiff = zPos-lmZ
            zHat            = np.array([[((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)+bias]],dtype=float)
            rhoHat          = ((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)
            Ht[0,ns*id+0]   = xDiff/rhoHat
            Ht[0,ns*id+1]   = yDiff/rhoHat
            Ht[0,ns*id+2]   = zDiff/rhoHat
            Ht[0,ns*id+bid] = 1.0

            timeDiff = abs(rhoTime-odomTime)
            if timeDiff < self.rhoTimeDiff:
                Rt        = np.array([[self.rhoSigmaLM**2]],dtype=float)
                mat       = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
                inv_mat   = np.linalg.inv(mat)
                Lt        = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
                self.pCov = np.dot(np.identity(ns*nv)-np.dot(Lt,Ht),self.pCov)
                self.xHat = np.add(self.xHat,np.dot(Lt,zt-zHat))

                print("Measurement update with range between",veh,"and",lmID)

    def intervehicleRangeMeasurementUpdate(self,veh1,veh2,msg):
        ns = self.NumStates
        nv = self.NumVeh
        if msg.range > 2.5:
            Ht      = np.zeros((1,ns*nv),dtype=float)
            zt      = np.array([[msg.range/1000.0]],dtype=float)
            rhoTime = (msg.header.stamp).to_sec()

            if veh1 == 'UAS1':
                odomTime = self.UAS1_Time
                id1      = 0
                if veh2 == 'UAS2':
                    id2 = 1
                    bid = 9
                if veh2 == 'UAS3':
                    id2 = 2
                    bid = 10
            if veh1 == 'UAS2':
                odomTime = self.UAS2_Time
                id1      = 1
                if veh2 == 'UAS1':
                    id2 = 0
                    bid = 9
                if veh2 == 'UAS3':
                    id2 = 2
                    bid = 10
            if veh1 == 'UAS3':
                odomTime = self.UAS3_Time
                id1      = 2
                if veh2 == 'UAS1':
                    id2 = 0
                    bid = 9
                if veh2 == 'UAS2':
                    id2 = 1
                    bid = 10
            
            x1_pos = self.xHat[ns*id1+0,0]
            y1_pos = self.xHat[ns*id1+1,0]
            z1_pos = self.xHat[ns*id1+2,0]

            x2_pos = self.xHat[ns*id2+0,0]
            y2_pos = self.xHat[ns*id2+1,0]
            z2_pos = self.xHat[ns*id2+2,0]

            bias   = self.xHat[ns*id1+bid,0]

            xDiff  = x1_pos - x2_pos
            yDiff  = y1_pos - y2_pos
            zDiff  = z1_pos - z2_pos

            zHat             = np.array([[((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)+bias]],dtype=float)
            rhoHat           = ((xDiff)**2+(yDiff)**2+(zDiff)**2)**(0.5)
            Ht[0,ns*id1+0]   = xDiff/rhoHat
            Ht[0,ns*id1+1]   = yDiff/rhoHat
            Ht[0,ns*id1+2]   = zDiff/rhoHat
            Ht[0,ns*id2+0]   = -xDiff/rhoHat
            Ht[0,ns*id2+1]   = -yDiff/rhoHat
            Ht[0,ns*id2+2]   = -zDiff/rhoHat
            Ht[0,ns*id1+bid] = 1.0

            timeDiff = abs(rhoTime-odomTime)
            if timeDiff < self.rhoTimeDiff:
                Rt        = np.array([[self.rhoSigmav2v**2]],dtype=float)
                mat       = np.add(Rt,np.dot(Ht,np.dot(self.pCov,np.transpose(Ht))))
                inv_mat   = np.linalg.inv(mat)
                Lt        = np.dot(np.dot(self.pCov,np.transpose(Ht)),inv_mat)
                self.pCov = np.dot(np.identity(ns*nv)-np.dot(Lt,Ht),self.pCov)
                self.xHat = np.add(self.xHat,np.dot(Lt,zt-zHat))

                print("Measurement update with range between",veh1,"and",veh2)

    def calculateError(self,veh,msg):
        lat            = float(msg.latitude)
        lon            = float(msg.longitude)
        alt            = float(msg.altitude)
        pos            = utm.from_latlon(lat,lon)
        xPos,yPos,zPos = self.convert_to_local(pos[0],pos[1],alt)
        local_pos      = np.array([xPos,yPos,zPos],dtype=float)
        ns             = self.NumStates
        nv             = self.NumVeh

        if veh == 'UAS1':
            id  = 0
            val = 0
            yaw = self.UAS1_Yaw
        if veh == 'UAS2':
            id  = 1
            val = 0
            yaw = self.UAS2_Yaw
        if veh == 'UAS3':
            id  = 2
            val = 0
            yaw = self.UAS3_Yaw
        
        xEst   = self.xHat[ns*id+0,0]
        yEst   = self.xHat[ns*id+1,0]
        zEst   = self.xHat[ns*id+2,0]
        psiEst = self.wrap_to_pi(self.xHat[ns*id+3,0])

        err = np.array([xPos-xEst, yPos-yEst, zPos+val-zEst, (180.0/np.pi)*self.wrap_to_pi(yaw-psiEst)],dtype=float)
        pos_3sigma = np.array([3*(self.pCov[ns*id+0,ns*id+0])**(0.5),\
                               3*(self.pCov[ns*id+1,ns*id+1])**(0.5),\
                               3*(self.pCov[ns*id+2,ns*id+2])**(0.5),\
                               (180.0/np.pi)*3*(self.pCov[ns*id+3,ns*id+3])**(0.5)],dtype=float)
        neg_3sigma = - pos_3sigma
        local_pos[2] = local_pos[2]+val
        return err, pos_3sigma, neg_3sigma, local_pos, (180.0/np.pi)*self.wrap_to_pi(yaw)
    
    def PublishEstimate(self,veh,err,pos_3sigma,neg_3sigma,local_pos,yaw):
        ns = self.NumStates
        if veh == "UAS1":
            id  = 0
            gpsMsg = self.UAS1_GPSMsg
            odomMsg = self.UAS1_OdomMsg
            time = self.UAS1_Time
            val  = 0
        if veh == "UAS2":
            id  = 1
            gpsMsg = self.UAS2_GPSMsg
            odomMsg = self.UAS2_OdomMsg
            time = self.UAS2_Time
            val = 0
        if veh == "UAS3":
            id  = 2
            gpsMsg = self.UAS3_GPSMsg
            odomMsg = self.UAS3_OdomMsg
            time = self.UAS3_Time
            val =0

        estPub = localization_estimates()
        estPub.header = odomMsg.header
        estPub.header.stamp = rospy.Time.from_sec(time)
        estPub.pn        = self.xHat[ns*id+0,0]
        estPub.pe        = self.xHat[ns*id+1,0]
        estPub.pd        = self.xHat[ns*id+2,0]
        estPub.psi       = (180.0/np.pi)*self.wrap_to_pi(self.xHat[ns*id+3,0])
        estPub.b_rho_lm1 = self.xHat[ns*id+4,0]
        estPub.b_rho_lm2 = self.xHat[ns*id+5,0]
        estPub.b_rho_lm3 = self.xHat[ns*id+6,0]
        estPub.b_rho_lm4 = self.xHat[ns*id+7,0]
        estPub.b_rho_lm5 = self.xHat[ns*id+8,0]
        estPub.b_rho_v1  = self.xHat[ns*id+9,0]
        estPub.b_rho_v2  = self.xHat[ns*id+10,0]

        errPub = localization_error()
        errPub.header = odomMsg.header
        errPub.header.stamp = rospy.Time.from_sec(time)
        errPub.pn   = err[0]
        errPub.pe   = err[1]
        errPub.pd   = err[2]
        errPub.psi  = err[3]
        errPub.pos_3_sigma_pn   = pos_3sigma[0]
        errPub.pos_3_sigma_pe   = pos_3sigma[1]
        errPub.pos_3_sigma_pd   = pos_3sigma[2]
        errPub.pos_3_sigma_psi  = pos_3sigma[3]
        errPub.neg_3_sigma_pn   = neg_3sigma[0]
        errPub.neg_3_sigma_pe   = neg_3sigma[1]
        errPub.neg_3_sigma_pd   = neg_3sigma[2]
        errPub.neg_3_sigma_psi  = neg_3sigma[3]

        posPub = localization_estimates()
        posPub.header = gpsMsg.header
        posPub.header.stamp = gpsMsg.header.stamp
        posPub.pn = local_pos[0]
        posPub.pe = local_pos[1]
        posPub.pd = local_pos[2] + val
        posPub.psi = yaw

        return estPub, errPub, posPub
    
    def CooperativeLocalization(self):
        while not rospy.is_shutdown():
            # Set coordinates for local origin
            ns = self.NumStates
            nv = self.NumVeh
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
                rospy.loginfo("GPS lock not received. Node not initialized")
                self.rate.sleep()
                continue
            else:
                # Update Odometry measurements as they are received
                if self.UAS1_OdomMsg is not None:
                    # Update UAS1 Time
                    self.UAS1_Time = (self.UAS1_OdomMsg.header.stamp).to_sec()
                    # Linear Velocity
                    self.UAS1_VelX = self.UAS1_OdomMsg.twist.twist.linear.x
                    self.UAS1_VelY = self.UAS1_OdomMsg.twist.twist.linear.y
                    self.UAS1_VelZ = self.UAS1_OdomMsg.twist.twist.linear.z
                    # Angular Velocity
                    self.UAS1_P    = self.UAS1_OdomMsg.twist.twist.angular.x
                    self.UAS1_Q    = self.UAS1_OdomMsg.twist.twist.angular.y
                    self.UAS1_R    = self.UAS1_OdomMsg.twist.twist.angular.z
                    # Orientation
                    quaternion      = self.UAS1_OdomMsg.pose.pose.orientation
                    self.UAS1_Roll, self.UAS1_Pitch, self.UAS1_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    if self.UAS1_FirstState == True:
                        id                    = 0
                        self.xHat[ns*id+0,0]  = self.UAS1_InitPos[0]
                        self.xHat[ns*id+1,0]  = self.UAS1_InitPos[1]
                        self.xHat[ns*id+2,0]  = self.UAS1_InitPos[2]
                        self.xHat[ns*id+3,0]  = (self.UAS1_Yaw)
                        self.UAS1_FirstState = False
                        self.currTime         = rospy.Time.now()
                        self.prevTime         = rospy.Time.now()
                if self.UAS2_OdomMsg is not None:
                    # Update UAS2 Time
                    self.UAS2_Time = (self.UAS2_OdomMsg.header.stamp).to_sec()
                    # Linear Velocity
                    self.UAS2_VelX = self.UAS2_OdomMsg.twist.twist.linear.x
                    self.UAS2_VelY = self.UAS2_OdomMsg.twist.twist.linear.y
                    self.UAS2_VelZ = self.UAS2_OdomMsg.twist.twist.linear.z
                    # Angular Velocity
                    self.UAS2_P    = self.UAS2_OdomMsg.twist.twist.angular.x
                    self.UAS2_Q    = self.UAS2_OdomMsg.twist.twist.angular.y
                    self.UAS2_R    = self.UAS2_OdomMsg.twist.twist.angular.z
                    # Orientation
                    quaternion      = self.UAS2_OdomMsg.pose.pose.orientation
                    self.UAS2_Roll, self.UAS2_Pitch, self.UAS2_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    if self.UAS2_FirstState == True:
                        id                     = 1
                        self.xHat[ns*id+0,0]   = self.UAS2_InitPos[0]
                        self.xHat[ns*id+1,0]   = self.UAS2_InitPos[1]
                        self.xHat[ns*id+2,0]   = self.UAS2_InitPos[2]
                        self.xHat[ns*id+3,0]   = (self.UAS2_Yaw)
                        self.UAS2_FirstState = False
                        self.currTime          = rospy.Time.now()
                        self.prevTime          = rospy.Time.now()
                if self.UAS3_OdomMsg is not None:
                    # Update UAS3 Time
                    self.UAS3_Time = (self.UAS3_OdomMsg.header.stamp).to_sec()
                    # Linear Velocity
                    self.UAS3_VelX = self.UAS3_OdomMsg.twist.twist.linear.x
                    self.UAS3_VelY = self.UAS3_OdomMsg.twist.twist.linear.y
                    self.UAS3_VelZ = self.UAS3_OdomMsg.twist.twist.linear.z
                    # Angular Velocity
                    self.UAS3_P    = self.UAS3_OdomMsg.twist.twist.angular.x
                    self.UAS3_Q    = self.UAS3_OdomMsg.twist.twist.angular.y
                    self.UAS3_R    = self.UAS3_OdomMsg.twist.twist.angular.z
                    # Orientation
                    quaternion       = self.UAS3_OdomMsg.pose.pose.orientation
                    self.UAS3_Roll, self.UAS3_Pitch, self.UAS3_Yaw = \
                        euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
                    if self.UAS3_FirstState == True:
                        id                     = 2
                        self.xHat[ns*id+0,0]   = self.UAS3_InitPos[0]
                        self.xHat[ns*id+1,0]   = self.UAS3_InitPos[1]
                        self.xHat[ns*id+2,0]   = self.UAS3_InitPos[2]
                        self.xHat[ns*id+3,0]   = (self.UAS3_Yaw)
                        self.UAS3_FirstState = False
                        self.currTime          = rospy.Time.now()
                        self.prevTime          = rospy.Time.now()

                if (self.UAS1_FirstState == False and self.UAS2_FirstState == False and self.UAS3_FirstState == False):
                    self.currTime = rospy.Time.now()
                    Ts = (self.currTime-self.prevTime).to_sec()
                    #print(Ts)
                    self.prevTime    = self.currTime
                    self.UAS1_Time  = self.UAS1_Time + Ts
                    self.UAS2_Time = self.UAS2_Time + Ts
                    self.UAS3_Time = self.UAS3_Time + Ts
                    self.prediction(Ts)

                    # GPS measurement update
                    if self.UAS1_GPSMsg is not None:
                        # Calculate error with respect to GPS measurements
                        UAS1_Err, UAS1_pos_3sigma, UAS1_neg_3sigma, UAS1_local_pos, UAS1_local_yaw = self.calculateError('UAS1',self.UAS1_GPSMsg)
                        if self.UAS1_GPSflag is True:
                            self.gpsMeasurementUpdate('UAS1',self.UAS1_GPSMsg)
                        else:
                            if self.heightFlag is True:
                                self.heightMeasurementUpdate('UAS1',self.UAS1_GPSMsg)
                    if self.UAS2_GPSMsg is not None:
                        # Calculate error with respect to GPS measurements
                        UAS2_Err, UAS2_pos_3sigma, UAS2_neg_3sigma, UAS2_local_pos, UAS2_local_yaw = self.calculateError('UAS2',self.UAS2_GPSMsg)
                        if self.UAS2_GPSflag is True:
                            self.gpsMeasurementUpdate('UAS2',self.UAS2_GPSMsg)
                        else:
                            if self.heightFlag is True:
                                self.heightMeasurementUpdate('UAS2',self.UAS2_GPSMsg)
                    if self.UAS3_GPSMsg is not None:
                        UAS3_Err, UAS3_pos_3sigma, UAS3_neg_3sigma, UAS3_local_pos, UAS3_local_yaw = self.calculateError('UAS3',self.UAS3_GPSMsg)
                        if self.UAS3_GPSflag is True:
                            self.gpsMeasurementUpdate('UAS3',self.UAS3_GPSMsg)
                        else:
                            if self.heightFlag is True:
                                self.heightMeasurementUpdate('UAS3',self.UAS3_GPSMsg)

                    # Landmark Measurement updates
                    if self.rangeflag is True:
                        if self.lm1_to_UAS1_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm1_to_UAS1_msg,self.lm1,'lm1')
                        if self.lm2_to_UAS1_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm2_to_UAS1_msg,self.lm2,'lm2')
                        if self.lm3_to_UAS1_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm3_to_UAS1_msg,self.lm3,'lm3')
                        if self.lm4_to_UAS1_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm4_to_UAS1_msg,self.lm4,'lm4')
                        if self.lm5_to_UAS1_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS1',self.lm5_to_UAS1_msg,self.lm5,'lm5')
                        if self.lm1_to_UAS2_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm1_to_UAS2_msg,self.lm1,'lm1')
                        if self.lm2_to_UAS2_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm2_to_UAS2_msg,self.lm2,'lm2')
                        if self.lm3_to_UAS2_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm3_to_UAS2_msg,self.lm3,'lm3')
                        if self.lm4_to_UAS2_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm4_to_UAS2_msg,self.lm4,'lm4')
                        if self.lm5_to_UAS2_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS2',self.lm5_to_UAS2_msg,self.lm5,'lm5')
                        if self.lm1_to_UAS3_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm1_to_UAS3_msg,self.lm1,'lm1')
                        if self.lm2_to_UAS3_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm2_to_UAS3_msg,self.lm2,'lm2')
                        if self.lm3_to_UAS3_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm3_to_UAS3_msg,self.lm3,'lm3')
                        if self.lm4_to_UAS3_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm4_to_UAS3_msg,self.lm4,'lm4')
                        if self.lm5_to_UAS3_msg is not None:
                            self.landmarkRangeMeasurementUpdate('UAS3',self.lm5_to_UAS3_msg,self.lm5,'lm5')

                    # Inter vehicle measurement updates
                    if self.coopflag is True:
                        if self.UAS2_to_UAS1_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS1','UAS2',self.UAS2_to_UAS1_msg)
                        if self.UAS3_to_UAS1_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS1','UAS3',self.UAS3_to_UAS1_msg)
                        if self.UAS1_to_UAS2_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS2','UAS1',self.UAS1_to_UAS2_msg)
                        if self.UAS3_to_UAS2_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS2','UAS3',self.UAS3_to_UAS2_msg)
                        if self.UAS1_to_UAS3_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS3','UAS1',self.UAS1_to_UAS3_msg)
                        if self.UAS2_to_UAS3_msg is not None:
                            self.intervehicleRangeMeasurementUpdate('UAS3','UAS2',self.UAS2_to_UAS3_msg)

                    ### Publish UAS1_ Estimates
                    UAS1_EstPub, UAS1_ErrPub, UAS1_TruePub = self.PublishEstimate('UAS1',UAS1_Err,UAS1_pos_3sigma,UAS1_neg_3sigma,UAS1_local_pos,UAS1_local_yaw)
                    self.UAS1_EstPub.publish(UAS1_EstPub)
                    self.UAS1_ErrPub.publish(UAS1_ErrPub)
                    self.UAS1_GPSPub.publish(UAS1_TruePub)
                    UAS2_EstPub, UAS2_ErrPub, UAS2_TruePub = self.PublishEstimate('UAS2',UAS2_Err,UAS2_pos_3sigma,UAS2_neg_3sigma,UAS2_local_pos,UAS2_local_yaw)
                    self.UAS2_EstPub.publish(UAS2_EstPub)
                    self.UAS2_ErrPub.publish(UAS2_ErrPub)
                    self.UAS2_GPSPub.publish(UAS2_TruePub)
                    UAS3_EstPub, UAS3_ErrPub, UAS3_TruePub = self.PublishEstimate('UAS3',UAS3_Err,UAS3_pos_3sigma,UAS3_neg_3sigma,UAS3_local_pos,UAS3_local_yaw)
                    self.UAS3_EstPub.publish(UAS3_EstPub)
                    self.UAS3_ErrPub.publish(UAS3_ErrPub)
                    self.UAS3_GPSPub.publish(UAS3_TruePub)

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
        qe.CooperativeLocalization()
    except rospy.ROSInterruptException():
        rospy.loginfo("Cannot publish estimate. Shutting down node")
        return
    
if __name__=="__main__":
    main()