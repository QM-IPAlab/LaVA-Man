import tarfile
import os

# 设置你的tar文件路径和要解压的目标路径
tar_path = '/data/home/acw694/CLIPort_new_loss/scratch/lava_man/data_sim/ravens_dataset_train.tar.gz'
extract_path = '/data/home/acw694/CLIPort_new_loss/scratch/lava_man/data_sim'

os.makedirs(extract_path, exist_ok=True)


with tarfile.open(tar_path, 'r') as tar:
    for member in tar:
        path_str = member.name

        if 'google' in path_str or 'full' in path_str:  
        # 获取文件的基本名（不含目录）
        # basename = os.path.basename(member.name)

        # 只提取以 '0000' 开头的文件
        #if basename.startswith('00'):
            tar.extract(member, path=extract_path)
            print(f"已解压: {member.name}")
