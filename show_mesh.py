
import trimesh


mesh = trimesh.load('/home/robot/Downloads/battery/battery_001/Scan/Scan.obj')

if mesh.visual.kind == 'texture':
    print("Texture loaded successfully.")
else:
    print("Texture did not load.")

target_faces = 10000  # Set your target number of faces

mesh.show()