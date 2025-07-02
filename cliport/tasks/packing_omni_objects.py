
"""Packing Google Objects tasks."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pandas as pd

import pybullet as p
TRAINING_DIR = 'omni_objs_processed'
INTER_CLASS_DIR = 'omni_objs_processed_inter_class'
INTRA_CLASS_CATEGORY = ['ball','book','bottle','boxed_beverage','bucket_noodle', 'cup', 'doll', 'donut', 'egg', 'garlic', 'hamburger', 'hat' ,
                        'mango', 'pear', 'pitaya', 'pizza', 'remote_control', 'rubik_cube', 'shoe', 'steamed_bun']

class PackingOmniObjects(Task):
    """Packing Seen Google Objects Group base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "pack {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.obj_path = TRAINING_DIR
        self.meta_data = self.get_object_metadata()
        

    def get_object_metadata(self):
        TEXT_DIR = "OmniObjects_text"
        metadata = []
        
        classes = sorted(os.listdir(self.obj_path))
        for category in classes:
            category_path = os.path.join(self.obj_path, category)
            
            instances = sorted(os.listdir(os.path.join(self.obj_path, category)))
            if category in INTRA_CLASS_CATEGORY:
                instances = instances[5:] # keep the first 5 instances for intra-class testing
            
            instances = instances[:5]
            for instance in instances:
                text_dir = os.path.join(TEXT_DIR, category, f"{instance}.txt")
                if not os.path.exists(text_dir):
                    print(f"Text directory does not exist: {text_dir}")
                    continue
                metadata.append([category, instance, text_dir])

        df_metadata = pd.DataFrame(metadata, columns=["class_name", "instance_name", "file_path"])
        return df_metadata
    
    def extract_summary(self, file_path):
        summary = None
        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()  # 读取第一行并去除首尾空格
            if first_line.startswith("Summary:"):
                summary = first_line[len("Summary:"):].strip().lower()  
                if summary.startswith("this is "):
                    summary = summary[len("this is "):].strip()
                if summary.startswith("it's "):
                    summary = summary[len("it's "):].strip()
        return summary

    def get_random_class(self):
        self.meta_data.sample(1)

    def get_random_instance(self, class_name):
        random_a_instances = self.meta_data.query("class_name == 'class_name'").sample(3)


    def reset(self, env):
        super().reset(env)

        # Add container box.
        zone_size = self.get_random_size(0.2, 0.35, 0.2, 0.35, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf): os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.08
        bboxes = []

        # Construct K-D Tree to roughly estimate how many objects can fit inside the box.
        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                      (size[split_axis] - 2 * min_object_dim) + \
                      node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        # Add Google Scanned Objects to scene.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        scale_factor = 7
        object_template = 'google/object-template.urdf'
        selected_objs, repeat_category  = self.choose_objects(len(bboxes))

        object_descs = []
        for i, bbox in enumerate(bboxes):
            size = bbox[3:] - bbox[:3]
            max_size = size.max()
            shape_size = max_size * scale_factor
            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                object_name = selected_objs[i][0]
                text_file = selected_objs[i][2]
                instance = selected_objs[i][1]
                obj_des = self.extract_summary(text_file)

                object_name_with_underscore = object_name.replace(" ", "_")
                #MARK here to change the obj mesh
                mesh_file = os.path.join(self.obj_path,
                                         object_name,
                                         instance,
                                         'Scan',
                                         'Scan_transformed.obj') 
                texture_file = os.path.join(self.obj_path,
                                            object_name,
                                            instance,
                                            'Scan',
                                            'Scan.jpg')
                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [shape_size, shape_size, shape_size],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(object_template, replace)
                    box_id = env.add_object(urdf, ps)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    object_ids.append((box_id, (0, None)))

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
                    object_points[box_id] = self.get_mesh_object_points(box_id)

                    object_descs.append(obj_des)
                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")
        
        try:
            self.set_goals(object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size)
            for i in range(480):
                p.stepSimulation()
        except KeyError as e:
            print(f"KeyError: {e}. Possible issue with object_id being None or missing in object_points.")
            print(f"Debug: object_ids={object_ids}")
            print(f"Debug: object_points.keys()={list(object_points.keys())}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def choose_objects(self, k):
        return self.choose_objects_different_classes(k)

    def choose_objects_different_classes(self, k):
        """
        从 DataFrame `df_metadata` 中随机选择 `k` 个不同类别的对象。
        
        参数:
            k (int): 需要选择的对象数量
            
        返回:
            list of tuples: 选中的 (class_name, instance_name, file_path)
        """
        # 确保每个类别只取一个对象
        unique_classes = self.meta_data["class_name"].unique()
        
        if k > len(unique_classes):
            raise ValueError(f"there's only {len(unique_classes)} classes")
        
        # 从类别中随机选择 k 个
        selected_classes = np.random.choice(unique_classes, k, replace=False)
        
        # 为每个类别随机选择一个对象
        selected_objects = []
        for class_name in selected_classes:
            object_row = self.meta_data[self.meta_data["class_name"] == class_name].sample(1).iloc[0]
            selected_objects.append((object_row["class_name"], object_row["instance_name"], object_row["file_path"]))
    
        return selected_objects, None


    def choose_objects_same_class(self, class_name, k):
        """
        从同一个类别中随机选择 k 个不同的对象。

        参数:
            class_name (str): 类别名称
            k (int): 需要选择的对象数量

        返回:
            list of tuples: 选中的 (class_name, instance_name, file_path)
        """
        # 获取该类别下的所有对象
        class_objects = self.meta_data[self.meta_data["class_name"] == class_name]

        if len(class_objects) < k:
            raise ValueError(f"类别 '{class_name}' 中只有 {len(class_objects)} 个对象，无法选择 {k} 个")

        # 随机选择 k 个对象
        selected_rows = class_objects.sample(k)
        
        selected_objects = [
            (row["class_name"], row["instance_name"], row["file_path"])
            for _, row in selected_rows.iterrows()
        ]

        return selected_objects, None


    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Random picking sequence.
        num_pack_objs = np.random.randint(1, len(object_ids))

        object_ids = object_ids[:num_pack_objs]
        true_poses = []
        for obj_idx, (object_id, _) in enumerate(object_ids):
            true_poses.append(zone_pose)

            chosen_obj_pts = dict()
            chosen_obj_pts[object_id] = object_points[object_id]

            self.goals.append(([(object_id, (0, None))], np.int32([[1]]), [zone_pose],
                               False, True, 'zone',
                               (chosen_obj_pts, [(zone_pose, zone_size)]),
                               1 / len(object_ids)))
            self.lang_goals.append(self.lang_template.format(obj=object_descs[obj_idx]))

        # Only mistake allowed.
        self.max_steps = len(object_ids)+1


class PackingOmniObjectsIntraClass(PackingOmniObjects):
    """Packing Google Objects Group unseen intra class objects."""

    def get_object_metadata(self):
            TEXT_DIR = "OmniObjects_text"
            metadata = []
            
            classes = sorted(os.listdir(self.obj_path))
            for category in classes:
                
                if category not in INTRA_CLASS_CATEGORY:
                    continue
                              
                instances = sorted(os.listdir(os.path.join(self.obj_path, category)))
                instances = instances[:5] # keep the first 5 instances for intra-class testing 

                for instance in instances:
                    text_dir = os.path.join(TEXT_DIR, category, f"{instance}.txt")
                    if not os.path.exists(text_dir):
                        print(f"Text directory does not exist: {text_dir}")
                        continue
                    metadata.append([category, instance, text_dir])

            df_metadata = pd.DataFrame(metadata, columns=["class_name", "instance_name", "file_path"])
            return df_metadata


class PackingOmniObjectsInterClass(PackingOmniObjects):
    """Packing Google Objects Group unseen intra class objects."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "pack {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.obj_path = INTER_CLASS_DIR
        self.meta_data = self.get_object_metadata()


class PackingOmniOjbectsGroup(PackingOmniObjects):
   
    def __init__(self):
        super().__init__()
        self.lang_template = "pack {obj} objects in the brown box"
        self.max_steps = 3
        self.unique_classes = self.meta_data["class_name"].unique()


    def choose_objects(self, k):
        return self.choose_objects_group_more_distractor(k)


    def choose_objects_group_one_distractor(self, k):
        # obtain unique classes
        if k > len(self.unique_classes):
            raise ValueError(f"There's only {len(self.unique_classes)} unique classes.")

        # choose one of the selected classes to repeat
        repeat_category, distractor_category = np.random.choice(self.unique_classes, 2, replace=False)

        # number of repeats
        num_repeats = np.random.randint(2, 4)
        num_distractors = k - num_repeats
        
        # 为每个类别随机选择一个对象
        selected_objects = []
        object_row = self.meta_data[self.meta_data["class_name"] == repeat_category].sample(1).iloc[0]
        for _ in range(num_repeats):
            selected_objects.append((object_row["class_name"], object_row["instance_name"], object_row["file_path"]))
    
        object_row = self.meta_data[self.meta_data["class_name"] == distractor_category].sample(1).iloc[0]
        for _ in range(num_distractors):   
            selected_objects.append((object_row["class_name"], object_row["instance_name"], object_row["file_path"]))

        return selected_objects, repeat_category


    def choose_objects_group_more_distractor(self, k):
        # obtain unique classes
        if k > len(self.unique_classes):
            raise ValueError(f"There's only {len(self.unique_classes)} unique classes.")

        # choose the category to repeat
        repeat_category = np.random.choice(self.unique_classes)

        # number of repeats
        num_repeats = np.random.randint(2, 4)
        num_distractors = k - num_repeats

        selected_objects = []
        # select repeat objects
        object_row = self.meta_data[self.meta_data["class_name"] == repeat_category].sample(1).iloc[0]
        for _ in range(num_repeats):
            selected_objects.append((object_row["class_name"], object_row["instance_name"], object_row["file_path"]))

        # decide how many distractor categories to use
        distractor_categories = [c for c in self.unique_classes if c != repeat_category]
        if num_distractors < 4:
            num_distractor_classes = 1
        else:
            num_distractor_classes = 2  
        
        distractor_categories = np.random.choice(distractor_categories, num_distractor_classes, replace=False)
        counts = [num_distractors // num_distractor_classes] * num_distractor_classes
        for i in range(num_distractors % num_distractor_classes):
            counts[i] += 1

        for category, count in zip(distractor_categories, counts):
            object_row = self.meta_data[self.meta_data["class_name"] == category].sample(1).iloc[0]
            for _ in range(count):
                selected_objects.append((object_row["class_name"], object_row["instance_name"], object_row["file_path"]))
    
        return selected_objects, repeat_category

  
    def set_goals(self, object_descs, object_names, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Pack all objects of the chosen (repeat) category.
        
        num_pack_objs = object_names.count(repeat_category)
        true_poses = []

        chosen_obj_pts = dict()
        chosen_obj_ids = []
        for obj_idx, (object_id, info) in enumerate(object_ids):
            if object_names[obj_idx] == repeat_category:
                true_poses.append(zone_pose)
                chosen_obj_pts[object_id] = object_points[object_id]
                chosen_obj_ids.append((object_id, info))

        self.goals.append((
            chosen_obj_ids, np.eye(len(chosen_obj_ids)), true_poses, False, True, 'zone',
            (chosen_obj_pts, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(obj=repeat_category))

        # Only one mistake allowed.
        self.max_steps = num_pack_objs+1


    def reset(self, env):
        Task.reset(self, env)

        # Add container box.
        zone_size = self.get_random_size(0.2, 0.35, 0.2, 0.35, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf): os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.08
        bboxes = []

        # Construct K-D Tree to roughly estimate how many objects can fit inside the box.
        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                      (size[split_axis] - 2 * min_object_dim) + \
                      node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        # Add Google Scanned Objects to scene.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        scale_factor = 7
        object_template = 'google/object-template.urdf'
        selected_objs, repeat_category  = self.choose_objects(len(bboxes))

        object_descs = []
        object_names = []
        for i, bbox in enumerate(bboxes):
            size = bbox[3:] - bbox[:3]
            max_size = size.max()
            shape_size = max_size * scale_factor
            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                object_name = selected_objs[i][0]
                text_file = selected_objs[i][2]
                instance = selected_objs[i][1]
                obj_des = self.extract_summary(text_file)

                object_name_with_underscore = object_name.replace(" ", "_")
                #MARK here to change the obj mesh
                mesh_file = os.path.join(self.obj_path,
                                         object_name,
                                         instance,
                                         'Scan',
                                         'Scan_transformed.obj') 
                texture_file = os.path.join(self.obj_path,
                                            object_name,
                                            instance,
                                            'Scan',
                                            'Scan.jpg')
                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [shape_size, shape_size, shape_size],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(object_template, replace)
                    box_id = env.add_object(urdf, ps)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    object_ids.append((box_id, (0, None)))
                    object_names.append(object_name)

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
                    object_points[box_id] = self.get_mesh_object_points(box_id)

                    object_descs.append(obj_des)
                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")
        
        try:
            self.set_goals(object_descs, object_names ,object_ids, object_points, repeat_category, zone_pose, zone_size)
            for i in range(480):
                p.stepSimulation()
        except KeyError as e:
            print(f"KeyError: {e}. Possible issue with object_id being None or missing in object_points.")
            print(f"Debug: object_ids={object_ids}")
            print(f"Debug: object_points.keys()={list(object_points.keys())}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False



class PackingOmniOjbectsGroupInter(PackingOmniOjbectsGroup):
     
    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "pack {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.obj_path = INTER_CLASS_DIR
        self.meta_data = self.get_object_metadata()
        self.unique_classes = self.meta_data["class_name"].unique()


class PackingOmniOjbectsGroupIntra(PackingOmniOjbectsGroup):
    
    def get_object_metadata(self):
        TEXT_DIR = "OmniObjects_text"
        metadata = []
        
        classes = sorted(os.listdir(self.obj_path))
        for category in classes:
            
            if category not in INTRA_CLASS_CATEGORY:
                continue
                            
            instances = sorted(os.listdir(os.path.join(self.obj_path, category)))
            instances = instances[:5] # keep the first 5 instances for intra-class testing 

            for instance in instances:
                text_dir = os.path.join(TEXT_DIR, category, f"{instance}.txt")
                if not os.path.exists(text_dir):
                    print(f"Text directory does not exist: {text_dir}")
                    continue
                metadata.append([category, instance, text_dir])

        df_metadata = pd.DataFrame(metadata, columns=["class_name", "instance_name", "file_path"])
        return df_metadata