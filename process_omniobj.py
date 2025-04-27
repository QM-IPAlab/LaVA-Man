"""
blender --background --python process_omniobj.py

"""

import os
import tarfile
import subprocess
import shutil
import bpy
import mathutils

# 定义原始扫描文件的目录
RAW_SCANS_DIR = "/home/robot/Repositories_chaoran/CLIPort_new_loss/omni_objs"
OUPUT_SCANS_DIR = "/media/robot/New Volume/datasets/OminiObj_processed"

# 获取所有 tar.gz 文件
tar_files = [f for f in os.listdir(RAW_SCANS_DIR) if f.endswith(".tar.gz")]
tar_files.sort()

if not tar_files:
    print("did no find .tar.gz files, quit.")
    exit(0)

# 进度条：遍历 tar.gz 文件
for n_zip_files, file in enumerate(tar_files):
    file_path = os.path.join(RAW_SCANS_DIR, file)
    
    # 解压到anotehr文件夹
    extract_dir = os.path.join(OUPUT_SCANS_DIR, file.replace(".tar.gz", ""))
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    print(f"\n unzip : {file_path} -> {extract_dir}")
    with tarfile.open(file_path, "r:gz") as tar:
        folders = []
        for member in tar.getmembers():
            if len(member.name.split("/")) == 2:
                folders.append(member.name.split("/")[1])
        folders = sorted(set(folders))
        selected_folders = folders[10:30] 

        members_to_extract = [m for m in tar.getmembers() if any(m.name.startswith(f'./{folder}') for folder in selected_folders)]
        tar.extractall(extract_dir, members=members_to_extract)

    # 找到所有的 Scan 目录
    scan_dirs = []
    all_sub_dirs = os.listdir(extract_dir)
    all_sub_dirs.sort()
    for sub_dir in all_sub_dirs:  # 遍历解压出来的子文件夹
        scan_path = os.path.join(extract_dir, sub_dir, "Scan")
        if os.path.isdir(scan_path):
            scan_dirs.append(scan_path)

    # 如果没有找到 Scan 目录，跳过此文件
    if not scan_dirs:
        print(f"warning: {extract_dir} no Scan dir !")
        continue

    # 进度条：遍历 Scan 目录
    for n_sub_fils, scan_dir in enumerate(scan_dirs):
        obj_file_path = os.path.join(scan_dir, "Scan.obj")
        output_path = os.path.join(scan_dir, "Scan_transformed.obj")
        scale_factor = 0.001  # 将模型缩小 1000 倍

        if not os.path.exists(obj_file_path):
            print(f"错误: 找不到 {obj_file_path}，跳过")
            continue

       # === 清空场景 ===
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # === 导入 OBJ 文件 ===
        bpy.ops.wm.obj_import(filepath=obj_file_path)

        # 获取导入的对象（假定 OBJ 文件只导入了一个物体）
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        # === 计算 COM 并调整到几何中心 ===

        # 确保处于 Object 模式
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # === 缩小 1000 倍 ===
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(location=True, scale=True)


        # === reduce the number of faces ===
        mesh = obj.data
        target_max_faces = 20000  # Set your target number of faces
        initial_face_count = len(mesh.polygons)

        if initial_face_count > target_max_faces:
            # 使用 Decimate Modifier 减少面数
            modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            modifier.ratio = target_max_faces / initial_face_count
            bpy.ops.object.modifier_apply(modifier="Decimate")


        # 计算所有顶点的平均位置（几何中心）

        vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
        geo_center = mathutils.Vector((
            sum(v.x for v in vertices) / len(vertices),
            sum(v.y for v in vertices) / len(vertices),
            sum(v.z for v in vertices) / len(vertices)
        ))

        # 平移：将 COM 归零
        obj.location -= geo_center

        # === 导出处理后的模型 ===
        bpy.ops.wm.obj_export(filepath=output_path)

        if os.path.exists(obj_file_path):
            os.remove(obj_file_path)

        mtl_file_path = obj_file_path.replace(".obj", ".mtl")
        if os.path.exists(mtl_file_path):
            os.remove(mtl_file_path)

        print(f"finished {n_sub_fils+1}/{len(scan_dirs)} in {n_zip_files+1}/{len(tar_files)}")  


print("Everything done.")
