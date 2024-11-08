"""
Run this script to convert the ROS Camera message to OpenCV numpy messge
This script needs to be ran in python2 !
"""

import cv2
import time
import rospy
import numpy as np

from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
cvBridge = CvBridge()

from corsmal_benchmark_s2.msg import Int16 # init in the work space

class ImageMsgConvertor:
    """Convert the camera raw msg to cv2 numpy mes for cv model"""

    def __init__(self):
    
        self.cam3_image_msg = None
        
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/camera3/color/image_raw", Image, self.cv2_img_callbck)
        
        self.pub = rospy.Publisher("/observation", numpy_msg(Int16), queue_size=10)

    def cv2_img_callbck(self, data):
        """Convert the camera raw message to cv2 bgr8"""
        self.cam1_image = cvBridge.imgmsg_to_cv2(data, 'bgr8')
    

    def process(self):
        """Receive image from cam3, convert and publish, and show"""
        
        time.sleep(2.0) # wait for rgb image messages
        rate = rospy.Rate(30) # 30hz

        print("processing... ")
    
        while not rospy.is_shutdown():
            
            # segment red objects in the image
            img = self.cam1_image
            img = img[:640,:1280,:]
            img = img[::-1,:,:]
            img = img.transpose(1,0,2)
            img = cv2.resize(img, (160,320))


            #import pdb; pdb.set_trace()
            image_msg = np.asarray(img, dtype=np.int16).flatten()
            self.pub.publish(image_msg)
            img = cv2.resize(img, (400,800))
            cv2.imshow('cam1',img)
            

            cv2.waitKey(1)
            rate.sleep()

        rospy.spin()
       

if __name__ == '__main__':
    ci = ImageMsgConvertor()
    ci.process()