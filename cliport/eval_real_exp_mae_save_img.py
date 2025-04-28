"""
Scripts for evaluating the real-world experiments.
"""
import time

import rospy
import numpy as np
from cv_bridge import CvBridge
cvBridge = CvBridge()
import cliport.utils.visual_utils as vu
from PIL import Image
import models_lib
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from cliport.utils.utils import preprocess

import numpy as np
from collections import deque
from corsmal_benchmark_s2.msg import Int16 # init in the work space
import cv2

import sys
sys.path.append(f'cliport/mae')

MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]
TEST_PATH = '/home/robot_tutorial/chaoran/CLIPort_new_loss/output_mae_robot_lang_mix_v2_full/checkpoint-60.pth'
MODEL = 'mae_robot_lang'
CHECKPOINT = '/home/robot_tutorial/chaoran/CLIPort_new_loss/output_mae_robot_lang_mix_v2_full/checkpoint-60.pth'
device = 'cuda'

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
        self.observation_queue_depth.append(depth)
    
   
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
    
    def get_depth(self, wait_time=0.1, retries=10):
        """Return the most recent image from the queue, retrying if necessary."""
        for attempt in range(retries):
            if self.observation_queue_depth:
                print("Image received")
                return self.observation_queue_depth[-1]
            else:
                rospy.logwarn(f"No image available. Retry {attempt+1}/{retries}...")
                time.sleep(wait_time)

def get_fix_transform():
    trasform_fix = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return trasform_fix

def generate_token(text_processor, lang, device):
    if type(lang) is str:
        decoded_strings = [lang]
    else:
        decoded_strings = [s.decode('ascii') for s in lang]
    processed_lang = text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
    processed_lang = processed_lang.to(device)
    return processed_lang

def load_image_to_tensor(image_path):
    """
    Load an image from the specified path and convert it to a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The image converted to a PyTorch tensor.
    """
    
    MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
    STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]
    # Define the transform to convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])

    # Apply the transform to the image
    image_tensor = transform(image_path)

    # Optional: Move the tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')
        print('Tensor moved to GPU')
    
    return image_tensor

# load data and model
transform_train = get_fix_transform()
model = models_lib.__dict__[MODEL](norm_pix_loss=False)
model.to(device)
checkpoint = torch.load(CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model'])

from transformers import AutoTokenizer
text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# while True:

#     img_receiver = ImageReceiver()
#     img = img_receiver.get_image()

#     lang_goal = input("Please pring language goal:")
#     print(f"Received language: {lang_goal}")
#     if lang_goal is None: lang_goal = "Put the orange ball into the brown box" 
#     print("Processing...")

#     if lang_goal.lower() == 'q':
#         print("Exiting loop.")
#         break
    
#     # bgr to rgb
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     img_input = load_image_to_tensor(img)
#     img_input = img_input.unsqueeze(0)
#     #brg to rgb
#     lang_processed = generate_token(text_processor, lang_goal, device)

#     with torch.no_grad():
        
#         loss, predict, mask = model(img_input, img_input, None, None, lang_processed, mask_ratio=1.00)
#         predict = model.unpatchify(predict)
#         predict = predict.detach().cpu()
#         predict = predict[0]
#         predict = (predict - predict.min()
#                 ) / (predict.max() - predict.min())
#         predict = predict.permute(1, 2, 0).numpy()

#     plt.imshow(predict)
#     plt.show() 

#     cont = input("Press Enter to continue, or 'q' to quit: ")
#     if cont.lower() == 'q':
#         print("Exiting loop.")
#         break


# Initialize the time tracking variable
last_save_time = time.time()
import os
os.makedirs('saved_images', exist_ok=True)

while True:
    img_receiver = ImageReceiver()
    img = img_receiver.get_image()
    depth = img_receiver.get_depth()
    # sleep for a while

    cv2.imshow("Image Viewer", img)  # OpenCV uses BGR
    key = cv2.waitKey(1)
    
    if key == 13:
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        timestamp = int(current_time)
        
        save_path = f"saved_images/saved_image_{timestamp}.png"
        plt.imsave(save_path, img)  # Save the image
        
        save_path_depth = f"saved_images/saved_image_{timestamp}_depth.png"
        plt.imsave(save_path_depth, depth, cmap='gray')  # Save the depth image

    
        print(f"Image saved")
    
    elif  key == ord('q'):
        print("Exiting loop.")
        break

