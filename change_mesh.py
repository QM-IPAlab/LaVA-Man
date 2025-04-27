#blender --background --python change_mesh.py 

import bpy
import mathutils
import sys
import os

# === 获取命令行传入的 Scan 目录 ===
if len(sys.argv) < 5:
    print("缺少参数！请传入 Scan 目录路径")
    sys.exit(1)

# === 配置部分 ===
scan_dir = sys.argv[-1]  # 取最后一个参数
obj_file_path = os.path.join(scan_dir, "Scan.obj")
output_path = os.path.join(scan_dir, "Scan_transformed.obj")
scale_factor = 0.001  # 将模型缩小 1000 倍


if not os.path.exists(obj_file_path):
    print(f"Error: Can not found {obj_file_path}")
    sys.exit(1)

print(f"Processing  {obj_file_path} ...")

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
print(f"save output obj to: {output_path}")

if os.path.exists(obj_file_path):
    os.remove(obj_file_path)
    print(f"delete original OBJ : {obj_file_path}")

mtl_file_path = obj_file_path.replace(".obj", ".mtl")
if os.path.exists(mtl_file_path):
    os.remove(mtl_file_path)