
# import trimesh


# mesh = trimesh.load('/home/robot/Downloads/battery/battery_001/Scan/Scan.obj')

# if mesh.visual.kind == 'texture':
#     print("Texture loaded successfully.")
# else:
#     print("Texture did not load.")

# target_faces = 10000  # Set your target number of faces

# simplified_mesh = mesh.simplify_quadric_decimation(0.1)

# simplified_mesh.show()

# import trimesh
# import pymeshlab

# # 加载 OBJ 模型文件（如果模型包含多个网格，则返回一个 Scene 对象）
# mesh_obj = trimesh.load('/home/robot/Downloads/battery/battery_001/Scan/Scan.obj')

# # 导出为 glTF 文件
# # 注意：使用 '.gltf' 扩展名将生成 JSON 格式的 glTF 文件，
# # 如果希望导出为二进制 glTF（glb），则使用 '.glb' 扩展名。
# mesh_obj.export('/home/robot/Downloads/battery/battery_001/Scan/output.glb')


# ms = pymeshlab.MeshSet()

# ms.load_new_mesh('/home/robot/Downloads/battery/battery_001/Scan/output.glb')

# ms.show_polyscope()

# ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture',
#                 targetfacenum=10000,
#                 preservenormal=True,
#                 preserveboundary=True,
#                 optimalplacement=True,
#                 planarquadric=True)

# ms.show_polyscope()
# ms.save_current_mesh('/home/robot/Downloads/battery/battery_001/Scan/oupput.obj', save_textures=True)
#blender --background --python change_mesh.py 

import bpy
import mathutils

# === 配置部分 ===
obj_file_path = "/home/robot/Downloads/projector/projector_002/Scan/Scan.obj"  # 替换为你的 .obj 文件路径
scale_factor = 0.001  # 将模型缩小 1000 倍

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

# 计算所有顶点的平均位置（几何中心）
mesh = obj.data
vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
geo_center = mathutils.Vector((
    sum(v.x for v in vertices) / len(vertices),
    sum(v.y for v in vertices) / len(vertices),
    sum(v.z for v in vertices) / len(vertices)
))

print(f"几何中心: {geo_center}")

# 平移：将 COM 归零
obj.location -= geo_center

# === 缩小 1000 倍 ===
obj.scale = (scale_factor, scale_factor, scale_factor)
bpy.ops.object.transform_apply(location=True, scale=True)

print(f"模型已缩小 1000 倍，并调整 COM 以匹配几何中心。")

# === 导出处理后的模型 ===
output_path = "/home/robot/Downloads/projector/projector_002/Scan/Scan_transformed.obj"  # 替换为你的导出路径
bpy.ops.wm.obj_export(filepath=output_path)
print(f"处理后的模型已导出至: {output_path}")

