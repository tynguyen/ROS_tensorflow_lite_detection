#!/usr/bin/python
from cv_bridge import CvBridge, CvBridgeError
import rospy
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes 
from utils import timeit_ros
from sensor_msgs.msg import Image
import Queue as queue 
from tflite_ssd.tflite_ssd import SSDTflite
class SSDRos(object):
    def __init__(self):
        rospy.loginfo("[tflite_ssd_ros.py] Init SSD Ros model - Start !")
        self._bridge = CvBridge()
        self._read_params()
        # Define the SSD model 
        self.model = 1; 
        #ssd_model = SSDTflite(args.model_file, args.label_file)

        #ssd_model(img, output_name)

        self._init_topics()
        self.msg_queue = queue.Queue(maxsize=5)
        rospy.loginfo("[tflite_ssd_ros.py] Init SSD Ros model - Completed!")
    @staticmethod
    def _read_subscriber_param(name):
        """ reading subscriber parameters from launch or yaml files """
        topic = rospy.get_param("~subscriber/" + name + "/topic")
        queue_size = rospy.get_param("~subscriber/" + name + "/queue_size", 10)
        return topic, queue_size

    @staticmethod
    def _read_publisher_param(name):
        """ reading publisher parameters from launch or yaml files """
        topic = rospy.get_param("~publisher/" + name + "/topic")
        queue_size = rospy.get_param("~publisher/" + name + "/queue_size", 1)
        latch = rospy.get_param("~publisher/" + name + "/latch", False)
        return topic, queue_size, latch

    def _init_topics(self):
        """ This function is initializing node publisher and subscribers for the node """
        # Publisher
        topic, queue_size, latch = self._read_publisher_param("bounding_boxes")
        self._pub = rospy.Publisher(
            topic, BoundingBoxes, queue_size=queue_size, latch=latch
        )
        topic, queue_size, latch = self._read_publisher_param("image")
        self._pub_viz = rospy.Publisher(
            topic, Image, queue_size=queue_size, latch=latch
        )
        # Image Subscriber
        for i in range(self.num_cameras):
            topic, queue_size = self._read_subscriber_param("image" + str(i))
            self._image_sub = rospy.Subscriber(
                topic,
                Image,
                self._image_callback,
                queue_size=queue_size,
                buff_size=2 ** 24,
            )
        rospy.logdebug("[flite_ssd_ros] publishers and subsribers initialized")

    def _image_callback(self, msg):
        """ Main callback which is saving the last received image """
        if msg.header is not None:
            self.msg_queue.put(msg)
            rospy.logdebug("[flite_ssd_ros] image recieved")

    def _write_message(self, detection_results, boxes, scores, classes):
        """ populate output message with input header and bounding boxes information """
        if boxes is None:
            return None
        for box, score, category in zip(boxes, scores, classes):
            # Populate darknet message
            left, bottom, right, top = box
            detection_msg = BoundingBox()
            detection_msg.xmin = left
            detection_msg.xmax = right
            detection_msg.ymin = top
            detection_msg.ymax = bottom
            detection_msg.probability = score
            detection_msg.Class = category
            detection_results.bounding_boxes.append(detection_msg)
        return detection_results

    @timeit_ros
    def process_frame(self):
        """ Main function to process the frame and run the infererence """
        # Deque the next image msg
        current_msg = self.msg_queue.get()
        current_image = None
        # Convert to image to OpenCV format
        try:
            current_image = self._bridge.imgmsg_to_cv2(current_msg, "bgr8")
            rospy.logdebug("[flite_ssd_ros] image converted for processing")
        except CvBridgeError as e:
            rospy.logdebug("Failed to convert image %s", str(e))
        # Initialize detection results
        if current_image is not None:
            rospy.logdebug("[flite_ssd_ros] processing frame")
            boxes, classes, scores, visualization = self.model(current_image)
            detection_results = BoundingBoxes()
            detection_results.header = current_msg.header
            detection_results.image_header = current_msg.header
            # construct message
            self._write_message(detection_results, boxes, scores, classes)
            # send message
            try:
                rospy.logdebug("[flite_ssd_ros] publishing")
                self._pub.publish(detection_results)
                if self.publish_image:
                    self._pub_viz.publish(
                        self._bridge.cv2_to_imgmsg(visualization, "bgr8")
                    )
            except CvBridgeError as e:
                rospy.logdebug("[flite_ssd_ros] Failed to convert image %s", str(e))


    def _read_params(self):
        """ Reading parameters for YOLORos from launch or yaml files """
        self.publish_image = rospy.get_param("~publish_image", False)
        # default paths to weights from different sources
        self.weights_path = rospy.get_param("~weights_path", "./weights/")
        self.config_path = rospy.get_param("~config_path", "./config/")
        self.label_filename = rospy.get_param("~label_filename", "coco_labels.txt")
        # parameters of ssd detector
        self.ssd_type          = rospy.get_param("~ssd_type", "ssdv3-416")
        self.postprocessor_cfg = rospy.get_param(
            "~postprocessor_cfg", "ssd_postprocess_config.json"
        )
        self.obj_threshold = rospy.get_param("~obj_threshold", 0.6)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.3)
        # default cuda device
        self.cuda_device = rospy.get_param("~cuda_device", 0)
        self.num_cameras = rospy.get_param("~num_cam", 1)
        rospy.logdebug("[tflite_ssd_ros]: Number of cameras", self.num_cameras)
        rospy.logdebug("[tflite_ssd_ros] parameters read")
            

