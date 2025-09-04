from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import requests
import numpy as np
from PIL import Image

initialize_compilation_cache()

IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"

sample_fn = create_sample_fn("kvablack/susie")
image = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
image_out = sample_fn(image, "open the drawer")
import pdb; pdb.set_trace()
# to display the images if you're in a Jupyter notebook
display(Image.fromarray(image))
display(Image.fromarray(image_out))