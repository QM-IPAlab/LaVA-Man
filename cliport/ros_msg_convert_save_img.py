"""
Run this script to convert the ROS Camera message to OpenCV numpy messge
This script needs to be ran in python2 !
"""

import cv2
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
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
        rospy.Subscriber("/camera3/aligned_depth_to_color/image_raw", Image, self.callback1d)

        self.pub = rospy.Publisher("/observation", numpy_msg(Int16), queue_size=10)
        self.pubd = rospy.Publisher("/observation_depth", numpy_msg(Int16), queue_size=10)

    def cv2_img_callbck(self, data):
        """Convert the camera raw message to cv2 bgr8"""
        self.cam1_image = cvBridge.imgmsg_to_cv2(data, 'bgr8')

    def callback1d(self, data):
        self.cam1_depth = cvBridge.imgmsg_to_cv2(data, 'passthrough')
        self.cam1_depth_msg = self.cam1_depth.astype(np.int16)
    

    def process(self):
        """Receive image from cam3, convert and publish, and show"""
        
        time.sleep(1.0) # wait for rgb image messages
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
            
            depth = self.cam1_depth_msg # 720x1280
            depth = depth[:640,:1280]
            depth = depth[::-1,:]
            depth = depth.transpose(1,0)
            depth = cv2.resize(depth, (160,320), interpolation=cv2.INTER_NEAREST)
            depth_msg = depth.flatten()
            self.pubd.publish(depth_msg)
            cv2.imshow('cam1',img)

            key = cv2.waitKey(1)
            
            if key == 13:
    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                current_time = time.time()
                timestamp = int(current_time)
                
                save_path = "saved_images/saved_image_{}.png".format(timestamp)
                plt.imsave(save_path, img)  # Save the image
                
                save_path_depth = "saved_images/saved_image_{}_depth.png".format(timestamp)
                plt.imsave(save_path_depth, depth, cmap='gray')  # Save the depth image
            
                print("Image saved")
            
            elif  key == ord('q'):
                import sys
                sys.exit()
            
            else: 
                continue

            rate.sleep()

        rospy.spin()
       

if __name__ == '__main__':
    ci = ImageMsgConvertor()
    ci.process()