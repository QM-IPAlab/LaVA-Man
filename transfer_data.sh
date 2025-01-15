#!/bin/bash

# 设置私钥文件
PRIVATE_KEY="/data/home/acw694/private_key_jade"

# 设置远程主机用户名和地址
REMOTE_USER="cxz00-txk47"
REMOTE_HOST="jade2.hartree.stfc.ac.uk"

# 本地目录
LOCAL_DIR="/data/home/acw694/CLIPort_new_loss/scratch/Acdif"

# 需要复制的文件夹列表
FOLDERS=(
  "/data/home/acw694/PlayableVideoGeneration/bair_images_dataset_large"
)

# 需要单独处理的文件夹（只复制特定的 checkpoint 文件）
CHECKPOINT_FOLDERS=(
  "/jmain02/home/J2AD007/txk47/cxz00-txk47/DiffPort/exps_diffportcap_resize2s_renor"
  "/jmain02/home/J2AD007/txk47/cxz00-txk47/DiffPort/exps_diffportcap_resize2s_after"
  "/jmain02/home/J2AD007/txk47/cxz00-txk47/DiffPort/exps_real/real_images-diffportcap_resize2s-n100-train"
)

JSON_FOLDERS=(
  "/jmain02/home/J2AD007/txk47/cxz00-txk47/AcDif/exps/bair"
  "/jmain02/home/J2AD007/txk47/cxz00-txk47/AcDif/flow_large"
)

# 复制不包含特定 checkpoint 文件的文件夹
for folder in "${FOLDERS[@]}"; do
  echo "Copying folder: $folder"
  # 使用rsync递归复制整个文件夹的内容，不排除任何文件
  rsync -avz -e "ssh -i $PRIVATE_KEY" "$REMOTE_USER@$REMOTE_HOST:$folder" "$LOCAL_DIR/$(basename $folder)"
done

# # 复制包含 checkpoint 文件的文件夹，指定只复制某些 .ckpt 文件
# for folder in "${CHECKPOINT_FOLDERS[@]}"; do
#   echo "Copying specific checkpoint files from folder: $folder"
  
#   # 使用rsync复制指定的 checkpoint 文件
#   rsync -avz -e "ssh -i $PRIVATE_KEY" --include='*/' --include='best*.ckpt' --include='pick-best.ckpt' --include='place-best.ckpt' --exclude='*.ckpt' "$REMOTE_USER@$REMOTE_HOST:$folder/" "$LOCAL_DIR/$(basename $folder)"
# done

# 复制包含 checkpoint 文件的文件夹，指定只 .json 文件
# for folder in "${JSON_FOLDERS[@]}"; do
#   echo "Copying specific json files from folder: $folder"
  
#   # 使用rsync复制指定的 checkpoint 文件
#   rsync -avz -e "ssh -i $PRIVATE_KEY" --include='*/' --exclude='*.ckpt' "$REMOTE_USER@$REMOTE_HOST:$folder/" "$LOCAL_DIR/$(basename $folder)"
# done

echo "File transfer complete."
