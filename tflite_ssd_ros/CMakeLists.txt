cmake_minimum_required(VERSION 2.8.3)
project(tflite_ssd_ros)

find_package(catkin REQUIRED
    COMPONENTS
        cv_bridge 
        darknet_ros_msgs
)

catkin_package()

catkin_python_setup()
