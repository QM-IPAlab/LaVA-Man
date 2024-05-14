from matplotlib import pyplot as plt
import torch

def show_image(image, title=''):

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_mean = torch.tensor(clip_color_mean).view(1, 1, 3)
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]
    clip_color_std = torch.tensor(clip_color_std).view(1, 1, 3)

    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * clip_color_std + clip_color_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')

