import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 如果需要使用标准的变换
default_transform = transforms.Compose([
    transforms.Resize(256),          # 先调整大小，确保最小边大于224
    transforms.CenterCrop(224),      # 中心裁剪到224x224
    transforms.ToTensor(),           # 转换为Tensor
])

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, frame_offset=30, num_images=10):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集根目录
            json_file (str): 选择的序列JSON文件
            transform (callable, optional): 应用于图像的变换
            frame_offset (int): 生成图像对时的帧偏移量
            num_images (int): 每个子文件夹随机选择的图片数量
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_offset = frame_offset
        self.num_images = num_images

        # 加载JSON文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # 准备图像对列表
        self.image_pairs = self._prepare_image_pairs()

    def _prepare_image_pairs(self):
        image_pairs = []

        for category, folders in self.data.items():  # return key-value
            for folder, frames in folders.items():
                # 如果帧数量少于需要的随机数量，选择全部，否则随机选择指定数量
                if len(frames) < self.num_images:
                    selected_frames = frames
                else:
                    selected_frames = random.sample(frames, self.num_images)

                # 为每个选择的帧生成图像对
                for frame in selected_frames:
                    pair_frame = self._get_pair_frame(frame, frames)
                    if pair_frame is not None:
                        img1_path = os.path.join(self.root_dir, category, folder, 'images', f'frame{frame:06d}.jpg')
                        img2_path = os.path.join(self.root_dir, category, folder, 'images', f'frame{pair_frame:06d}.jpg')
                        image_pairs.append((img1_path, img2_path))

        return image_pairs

    def _get_pair_frame(self, frame, frames):
        """根据给定帧生成±30范围内的帧 确保不越界。"""
        valid_frames = [f for f in frames if abs(f - frame) <= self.frame_offset and f != frame]
        if valid_frames:
            return random.choice(valid_frames)
        return None

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


if __name__ == "__main__":
    # 示例用法
    dataset = ImagePairDataset(
        root_dir='../datasets/co3d',  # 数据集根目录
        json_file='../datasets/co3d/selected_seqs_train.json',  # JSON文件路径
        transform=default_transform  # 图像转换
    )

    # 检查数据集是否正确加载
    for img1, img2 in dataset:
        print(img1.shape, img2.shape)
