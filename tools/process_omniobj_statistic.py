"""
blender --background --python process_omniobj.py

"""

import os
import bpy
import mathutils
import math


N_INSTANCES_PROCESS = 30
N_CLASS_PROCESS = 216        # read the file
OUPUT_SCANS_DIR = "/media/robot/New Volume/datasets/OminiObj_processed"
SAVE_DIR = "/home/robot/Repositories_chaoran/CLIPort_new_loss/omni_objs_images"
CSV_PATH = os.path.join(SAVE_DIR, "class_statistics2.csv")

all_files = os.listdir(OUPUT_SCANS_DIR)
all_files.sort()

with open(CSV_PATH, "w") as f:
    f.write("class_name,num_instances,dimensions,example_image_path\n")

# process each class
for n, class_name in enumerate(all_files):
    if n > N_CLASS_PROCESS:
        break
    class_dir = os.path.join(OUPUT_SCANS_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    print(f"Processing {class_name}")
    all_files = os.listdir(class_dir)
    all_files.sort()
    num_of_instances = len(all_files)
    print(f"Number of instances: {num_of_instances}")
    
    # get the dimensions of the first file
    for n, file_name in enumerate(all_files):
        
        if n >= N_INSTANCES_PROCESS:
            break
        file_path = os.path.join(class_dir, file_name, 'Scan', 'Scan_transformed.obj')
        if not os.path.isfile(file_path):
            continue

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.context.scene.render.film_transparent = True


        # === 导入 OBJ 文件 ===
        bpy.ops.wm.obj_import(filepath=file_path)
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        # === 确保处于 Object 模式
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # === 获取 mesh 顶点，计算尺寸 ===
        mesh = obj.data
        vertices = [obj.matrix_world @ v.co for v in mesh.vertices]

        bbox_min = mathutils.Vector((min(v.x for v in vertices),
                                    min(v.y for v in vertices),
                                    min(v.z for v in vertices)))

        bbox_max = mathutils.Vector((max(v.x for v in vertices),
                                    max(v.y for v in vertices),
                                    max(v.z for v in vertices)))

        size = bbox_max - bbox_min
        dimensions_str = f"{size.x:.4f}, {size.y:.4f}, {size.z:.4f}"

        # # === 设置摄像头和光源 ===
        # light_data = bpy.data.lights.new(name="Light", type='SUN')
        # light_object = bpy.data.objects.new(name="Light", object_data=light_data)
        # bpy.context.collection.objects.link(light_object)
        # light_object.location = (5.0, 5.0, 5.0)

        # # === 计算模型中心和大小 ===
        # bbox_center = (bbox_min + bbox_max) / 2
        # max_dim = max(size.x, size.y, size.z)
        # cam_distance = max_dim * 2.5

        # # === 相机位置（从斜上方看）===
        # cam_location = bbox_center + mathutils.Vector((cam_distance, cam_distance, cam_distance))

        # # 创建相机
        # cam_data = bpy.data.cameras.new(name="Camera")
        # cam_data.angle = math.radians(20)  # 缩小视野角，进一步放大效果
        # cam_obj = bpy.data.objects.new("Camera", cam_data)
        # bpy.context.collection.objects.link(cam_obj)

        # # 设置位置
        # cam_obj.location = cam_location

        # # 让相机对准模型中心
        # direction = bbox_center - cam_location
        # rot_quat = direction.to_track_quat('-Z', 'Y')  # Blender 摄像头默认朝 -Z
        # cam_obj.rotation_euler = rot_quat.to_euler()

        # # 启用 World 节点（如果还没启用）
        # bpy.context.scene.world.use_nodes = True

        # # 获取背景节点
        # bg_node = bpy.context.scene.world.node_tree.nodes["Background"]

        # # 设置强度（默认是 1.0，建议调到 2.0 - 5.0 之间）
        # bg_node.inputs[1].default_value =5.0

        # # 激活摄像头
        # bpy.context.scene.camera = cam_obj

        # # === 渲染图像 ===
        # # 设置渲染图像尺寸（256 x 256）
        # bpy.context.scene.render.resolution_x = 256
        # bpy.context.scene.render.resolution_y = 256
        # bpy.context.scene.render.resolution_percentage = 100

        # render_path = os.path.join(SAVE_DIR, f"{class_name}_{file_name}.png")
        # bpy.context.scene.render.image_settings.file_format = 'PNG'
        # bpy.context.scene.render.filepath = render_path
        # bpy.ops.render.render(write_still=True)

        # 结果可用于填表
        print(f"Dimensions: {dimensions_str}")
        #print(f"Image saved at: {render_path}")

        # add to the dataframe
    
        with open(CSV_PATH, "a") as f:
            f.write(f"{class_name},{num_of_instances},{file_name},{dimensions_str}\n")

        #import pdb; pdb.set_trace()