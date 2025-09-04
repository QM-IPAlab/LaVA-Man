from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import cv2

import asyncio

# Save original
_orig_cond_init = asyncio.Condition.__init__

def _patched_condition_init(self, lock=None, *args, **kwargs):
    if lock is not None:
        # Workaround for bpo-45416: ensure the lock's _loop matches
        getattr(lock, '_get_loop', lambda: None)()
    # Now call the real initializer
    return _orig_cond_init(self, lock=lock, *args, **kwargs)

# Apply the patch
asyncio.Condition.__init__ = _patched_condition_init

initialize_compilation_cache()

sample_fn = create_sample_fn("kvablack/susie")
# sample_fn = create_sample_fn(
#         "/home/robot/Repositories_chaoran/CLIPort_new_loss/checkpoints/susie_diffusion",
#         "kvablack/dlimp-diffusion/9n9ped8m",
#         num_timesteps=50,
#         prompt_w=7.5,
#         context_w=1.5,
#         eta=0.0
#         )

image = imageio.imread('/home/robot/Repositories_chaoran/CLIPort_new_loss/1_ori.png')
image = image[:, :, :3]  # Ensure the image has 3 channels (RGB)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
image = image.astype(np.float32)
image_out = sample_fn(image, "open the drawer")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.astype(np.uint8))
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_out)
plt.title("Output Image")
plt.axis('off')

plt.show()
import pdb; pdb.set_trace()


# to display the images if you're in a Jupyter notebook
display(Image.fromarray(image))
display(Image.fromarray(image_out))