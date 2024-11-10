import numpy as np
import cv2
import os
import torch


def tensor_to_cv2_img(tensor, to_rgb=False):
    """
    tensor: [1, channels, height, width]
    """
    # 将tensor从GPU移动到CPU并转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        img_np = tensor.cpu().detach().numpy()
    else:
        img_np = tensor
    # 调整通道的顺序

    if len(img_np.shape) == 4:
        img_np = img_np.squeeze()

    if img_np.shape[0] <= 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    # 如果数据范围是[0, 1]，则将其转换为[0, 255]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    # rgb->bgr 用于OpenCV显示
    if not to_rgb:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


def label_to_color_map(labels, num_clusters):
    np.random.seed(0)  # for consistency
    colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)
    return colors[labels]


def label_to_color_map_fixed(labels, num_clusters):
    # Define a fixed color map as a numpy array
    # Each row represents a color (R, G, B)
    # This is just an example, you can define your own color map
    colors = np.array([
        [128, 0, 0],  # 类别1
        [0, 128, 0],  # 类别2
        [128, 128, 0],  # 类别3
        [0, 0, 128],  # 类别4
        [128, 0, 128],  # 类别5
        [0, 128, 128],  # 类别6
        [128, 128, 128],  # 类别7
        [64, 0, 0],  # 类别8
        [192, 0, 0],  # 类别9
        [64, 128, 0]  # 类别10
    ], dtype=np.uint8)

    # Make sure the number of colors is at least the number of clusters
    assert colors.shape[0] >= num_clusters, "Not enough colors defined"

    return colors[labels]


def get_kmeans_features(latent, K=10):
    """
    Get the kmeans features (color mask) from the latent features.

    Args:
        latent: [channels, height, width]
        K: number of clusters
    Returns:
        color_mask: [height, width, 3]
    """
    # reshape
    channels, height, width = latent.shape
    latent = latent.transpose(1, 2, 0).reshape(-1, channels)  # [batch_size * height * width, channels]

    # kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(np.float32(latent), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    color_mask = label_to_color_map_fixed(labels.flatten(), K)
    color_mask = color_mask.reshape(height, width, 3)

    return color_mask


def save_feature_map(image, mask, folder_name, file_name=None, prompt=None):
    """
    Saves the feature map to a local file using OpenCV.

    Args:
        image (np.array): original image in RGB format.
        feature_map (np.array)
        folder_name (str) : Path to save the combined image.

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """

    color_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    height, width, _ = combined.shape

    total_width = 3 * width  # 假设三张图片并排放置

    # 创建一个空白背景图
    merged_image = np.zeros((height, total_width, 3), dtype=np.uint8)

    # 将三张图片复制到背景图上
    merged_image[:, :width] = image
    merged_image[:, width:2 * width] = color_mask
    merged_image[:, 2 * width:] = combined

    # 添加文字
    if prompt is not None:
        text = str(prompt)  # 将变量l转换为字符串
        position = (10, 50)  # 设置文字的开始位置
        cv2.putText(merged_image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    os.makedirs(f'{folder_name}', exist_ok=True)

    if file_name is None:
        exits = os.listdir(f'{folder_name}')
        ids = int(len(exits))
        file_name = f'{ids}'

    file_name = f"{folder_name}/{file_name}.png"
    is_saved = cv2.imwrite(file_name, merged_image)
    return is_saved

    # MARK - save tensor with heatmap
    # img = in_tensor.cpu().detach().numpy()
    # img = img[0,...]
    # img = img.transpose(1, 2, 0)

    # crop_img = crop.cpu().detach().numpy()
    # crop_img = crop_img[0,...]
    # crop_img = crop_img.transpose(1, 2, 0)

    # for i in range(3):
    #     save = utils.save_tensor_with_heatmap(img,key_out_one[:,i,...], f"vis_12_Dec/vis_trans_keyout1_{i}.png", l)
    #     save = utils.save_tensor_with_heatmap(img,key_out_two[:,i,...], f"vis_12_Dec/vis_trans_keyout2_{i}.png", l)
    #     save = utils.save_tensor_with_heatmap(img, logits[:,i,...], f"vis_12_Dec/vis_trans_logits_{i}.png", l)

    #     save = utils.save_tensor_with_heatmap(crop_img,query_out_one[0,i,...], f"vis_12_Dec/vis_trans_keyout1_crop{i}.png", "")
    #     save = utils.save_tensor_with_heatmap(crop_img,query_out_two[0,i,...], f"vis_12_Dec/vis_trans_keyout2_crop{i}.png", "")
    #     save = utils.save_tensor_with_heatmap(crop_img, kernel[0,i,...], f"vis_12_Dec/vis_trans_logits_crop{i}.png", "")


def save_tensor_with_heatmap(image: np.ndarray, heatmap: np.ndarray, filename, l=None, return_img=False):
    """
    Save an original image, its heatmap, and the overlay of both to a local file using OpenCV.
    """
    # Remove singleton dimensions
    # tensor = tensor.squeeze()

    # Ensure tensor is 2D after squeezing
    if len(heatmap.shape) != 2:
        raise ValueError("Tensor should be 2D for heatmap visualization after squeezing.")

    # Convert tensor to numpy array
    # img = original_img[:,:,:3]
    # data = tensor.detach().cpu().numpy()

    # Normalize data to 0-255 for heatmap
    data_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(data_normalized, cv2.COLORMAP_JET)

    # overlay heatmap on original image
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # get the index of the maximum value
    position = np.unravel_index(np.argmax(data_normalized), data_normalized.shape)
    # from icecream import ic
    # ic("heatmap position", position)

    # cv2.circle(overlay, (position[1], position[0]), 5, (0, 0, 255), 2)
    # concatenate original image, heatmap, and overlay
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    combined = np.hstack((image, heatmap, overlay))

    # Create a white line (padding area) for the text below the combined image
    padding_height = 50  # Height of the white line
    white_line = np.ones((padding_height, combined.shape[1], 3), dtype=np.uint8) * 255  # RGB white line

    # Concatenate the white line to the bottom of the combined image
    combined_with_line = np.vstack((combined, white_line))

    if l is not None:
        text = str(l)  # 将变量l转换为字符串
        position = (10, combined.shape[0] + 30)  # 设置文字的开始位置
        cv2.putText(combined_with_line, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if return_img:
        return combined_with_line
    else:
        #folder_name = os.path.dirname(filename)
        return cv2.imwrite(filename, combined_with_line)


def move_to_device(batch, device):
    # 活用递归函数
    if isinstance(batch, (list, tuple)):
        return [move_to_device(b, device) for b in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        # print warning and return original object
        print("Unable to move batch to device")
        return batch


def visulize_infeats_kmeans(infeats, image, prompt, K=10, folder_name='vis', img_name='infeats'):
    """
    visulize infeats and save them
    Args:
        infeats (list of tensor): list of 7 with dimensions of
            [1,320,64,64]
            [1,640,32,32]
            [1,1280,32,32]
            [1,1280,32,32]
            [1,1280,64,64]
            [1,640,128,128]
            [1,320,128,128]
        image (tensor): image tensor
    """
    for i, infeat in enumerate(infeats):

        if len(infeat.shape) == 4:
            infeat = infeat.squeeze(0)

        if isinstance(infeat, torch.Tensor):
            infeat = infeat.cpu().detach().numpy()

        color_mask = get_kmeans_features(infeat, K)

        is_saved = save_feature_map(image, color_mask, folder_name, f'{img_name}_{i}', prompt=prompt)
        print(f'save {is_saved}')


def show_mask_on_image(mask, img, l=None, output_path=None):
    """
    used for segmentor. Show heatmap on image

    Args:
        mask (tensor): mask tensor of shape [1,1,H,W]
        img (tensor): image tensor of shape [1,3,H,W] \in [0,1]
    """

    mask_save = mask.cpu().detach().numpy()
    mask_save = mask_save[0, 0, ...]
    img_save = img.cpu().detach().numpy()
    img_save = img_save[0, ...] * 255
    img_save = img_save.transpose(1, 2, 0)
    img_save = img_save.astype(np.uint8)

    img_save = cv2.resize(img_save, (mask_save.shape[0], mask_save.shape[1]), interpolation=cv2.INTER_NEAREST)
    idx = len(os.listdir(f'./{output_path}'))

    save_tensor_with_heatmap(img_save, mask_save, f"{output_path}/test{idx}.png", l=l)
