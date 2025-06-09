#!/usr/bin/env python3
import os
import shutil
import sys

def remove_masks(root_dir):
    """
    遍历 root_dir 下的所有子目录，
    删除所有名称为 "masks" 的目录。
    使用 bottom-up 遍历，确保先删除子目录。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == "depths":
                full_path = os.path.join(dirpath, dirname)
                print(f"删除目录：{full_path}")
                try:
                    shutil.rmtree(full_path)
                except Exception as e:
                    print(f"删除 {full_path} 时出错: {e}")

if __name__ == "__main__":
    # 如果通过命令行传入目录，则使用传入的目录，否则默认使用 "co3d"
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "scratch/co3d"
    
    if not os.path.exists(root_dir):
        print(f"目录 {root_dir} 不存在")
        sys.exit(1)
    
    remove_masks(root_dir)
