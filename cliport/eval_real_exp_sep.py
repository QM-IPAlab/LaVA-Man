"""
Scripts for evaluating the real-world experiments.
"""
import cv2
import time

import rospy
import numpy as np
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
cvBridge = CvBridge()
import matplotlib.pyplot as plt
import cliport.utils.visual_utils as vu
from geometry_msgs.msg import Point
from cliport.eval_sep import CkptManager, get_model_path
import numpy as np
import hydra
from collections import deque
from cliport import agents
from cliport.utils import utils
import torch

from corsmal_benchmark_s2.msg import Int16 # init in the work space

class ImageReceiver:
    def __init__(self) -> None:
        self.observation_queue = deque(maxlen=10)  # Circular buffer to store the last 10 images
        self.observation_queue_depth = deque(maxlen=10)  # Circular buffer to store the last 10 images
        self.received = False  # Flag to check if at least one image has been received

        rospy.init_node('listener', anonymous=True)
        self.subscriber = rospy.Subscriber("/observation", Int16, self.callback)
        self.subscriber_depth = rospy.Subscriber("/observation_depth", Int16, self.callback1d)
        
    def callback(self, data):
        """Receive images from /observation and store in a queue."""
        bgr = np.reshape(data.data, (320, 160, 3))
        bgr = bgr.astype(np.uint8)
        self.observation_queue.append(bgr)

    def callback1d(self, data):
        """Receive images from /observation_depth and store in a queue."""
        depth = np.reshape(data.data, (320, 160))
        depth = depth.astype(np.uint8)
        self.observation_queue.append(depth)
    
   
    def get_image(self, wait_time=0.1, retries=10):
        """Return the most recent image from the queue, retrying if necessary."""
        for attempt in range(retries):
            if self.observation_queue:
                print("Image received")
                return self.observation_queue[-1]  # Return the most recent image
            else:
                rospy.logwarn(f"No image available. Retry {attempt+1}/{retries}...")
                time.sleep(wait_time)  # Wait before retrying

        rospy.logerr("No image received after max retries.")
        return None


def show_image(image):
    """Display an image using matplotlib."""
    plt.figure()
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()


@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):

    # Load configs
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    tcfg['train']['exp_folder'] = vcfg['exp_folder']
    eval_task = vcfg['eval_task']
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])
    
    # load agent
    tcfg['pretrain_path'] = None
    tcfg['train']['batchnorm'] = True
    agent = agents.names[vcfg['agent']](name, tcfg, None, None, 'both')
    agent.eval()

    # Load checkpoint
    ckpt_manager = CkptManager(vcfg)
    eval_pick, eval_place = ckpt_manager.get_test_ckpt()

    existing_results = {}
    model_pick, model_place, test_name, to_eval = get_model_path(
            eval_pick[0], eval_place[0], vcfg, existing_results)
    agent.load_sep(model_pick, model_place)
    print(f"Loaded: {eval_pick[0]}, {eval_place[0]}")

    # run prediction
    img = []  # make it [320, 160, 3]
    lang_goal = []
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    while not input('Press q to exit') == "q":

        # read the numpy image (from observation)
        img_receiver = ImageReceiver()
        img = img_receiver.get_image()
        
        lang_goal = input("Please pring language goal:")
        print(f"Received language: {lang_goal}")
        if lang_goal is None: lang_goal = "Put the black mouse into the white plate" 
        print("Processing...")

        img_t = torch.tensor(img).unsqueeze(0).to(device_type).float()
        # Attention model forward pass.
        pick_inp = {'inp_img': img_t, 'lang_goal': lang_goal}
        pick_conf = agent.attn_forward(pick_inp)
        pick_conf = pick_conf.squeeze(0)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
        p0_pix_t = torch.tensor(p0_pix).unsqueeze(0).to(device_type).float()

        # Transport model forward pass.
        place_inp = {'inp_img': img_t, 'p0': p0_pix_t, 'lang_goal': lang_goal}
        place_conf = agent.trans_forward(place_inp)
        place_conf = place_conf.squeeze(0)

        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        pick_conf = pick_conf[:,:,0]
        place_conf = place_conf[:,:, argmax[2]]
        heatmap_pick = vu.save_tensor_with_heatmap(img, pick_conf, l=lang_goal, return_img=True)
        show_image(heatmap_pick[:,:,::-1])
        heatmap_place = vu.save_tensor_with_heatmap(img, place_conf, l=lang_goal, return_img=True)
        show_image(heatmap_place[:,:,::-1])
        
        # Publish point
    
    pub_pick = rospy.Publisher("/pick", Point, queue_size=10)
    pub_place = rospy.Publisher("/place", Point, queue_size=10)
    rate = rospy.Rate(10)
    print("Publishing point")
    print("pick:", p0_pix[0], p0_pix[1])
    print("place:", p1_pix[0], p1_pix[1])
    
    while not rospy.is_shutdown():
        pub_pick.publish(p0_pix[0], p0_pix[1], 0)
        pub_place.publish(p1_pix[0], p1_pix[1], p1_theta)
        rate.sleep()


        


if __name__ == '__main__':
    main()
