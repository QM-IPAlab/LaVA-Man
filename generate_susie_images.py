from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import requests
import numpy as np
from PIL import Image
import os
import json
import cv2
import tqdm
initialize_compilation_cache()

#IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"

# img_dir="/media/robot/New Volume/temp_backup/real_annotated"
img_dir="/media/robot/New Volume/temp_backup/real_img_eng207"

sample_fn = create_sample_fn("kvablack/susie")
# image = np.array(Image.open('/home/robot/Repositories_chaoran/CLIPort_new_loss/1_ori.png').resize((256, 256)))

# read annotations from a JSON file
with open(os.path.join(img_dir, "annotations.json"), 'r') as f:
    annotations = json.load(f)

# read image from file
selected_images = []
for fn in os.listdir(img_dir):
    if not fn.endswith(".png") or "depth" in fn or "goal" in fn or not("Sample" in fn):
        continue
    selected_images.append(fn)

for fn in tqdm.tqdm(selected_images):
    instruction = annotations[fn.replace(".png", "")]["instruction"]
    print(f"Processing {fn} with instruction: {instruction}")
    img_orig = Image.open(os.path.join(img_dir, fn))
    image = np.array(img_orig.resize((256, 256)))
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = image[:, :, :3]  # Remove the alpha channel
    image_out = sample_fn(image, instruction)
    image_out = sample_fn(image_out, instruction)
    image_out = sample_fn(image_out, instruction)
    image_out = sample_fn(image_out, instruction)
    # reshape to original size
    image_out = cv2.resize(image_out, (img_orig.width, img_orig.height))
    image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(img_dir, "{}_goal.png".format(fn.replace(".png", ""))), image_out)

print("Completed ")

# image = image[:, :, :3]

# to display the images if you're in a Jupyter notebook
#display(Image.fromarray(image))
#display(Image.fromarray(image_out))

print(image_out.shape)

Image.fromarray(image_out.astype(np.uint8)).save("output1.png")
