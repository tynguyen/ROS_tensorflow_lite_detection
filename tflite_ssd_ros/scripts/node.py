#!/usr/bin/python

import rospy
from tflite_ssd_ros.tflite_ssd_ros import SSDRos 
import pdb

if __name__ == "__main__":
    rospy.init_node("tflite_ssd_ros", log_level=rospy.INFO)
    rospy.loginfo("[tflite_ssd_ros] starting the node")
    try:
        network = SSDRos()
    except rospy.ROSInterruptException:
        pass
    publish_rate = rospy.get_param("~publish_rate", 10)
    sleep_time = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():
        network.process_frame()
        sleep_time.sleep()
        rospy.loginfo("[tflite_ssd_ros node] Process a frame")
