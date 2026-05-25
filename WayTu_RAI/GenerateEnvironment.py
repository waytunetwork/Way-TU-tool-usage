import robotic as ry
import numpy as np
import random
import os
import time
import csv
import json
import WayTu_RAI.model_utils as mutils
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt

from WayTu_RAI.GenerateTools import GenerateTools
from WayTu_RAI.PrimitiveTools import GeneratePrimitiveTools
from WayTu_RAI.DistractorTools import GenerateDistractiveTools
from WayTu_RAI.RealisticTools import GenerateRealisticTools
from WayTu_RAI.GrillingEnvironment import GrillingEnvironment
from WayTu_RAI.Environments.LiftingEnvironment import LiftingEnvironment
from WayTu_RAI.CameraRAI import CameraRAI
from WayTu_RAI.Heuristic import Heuristic

from WayTu_RAI.Environments.MinigolfEnvironment import MinigolfEnvironment
from WayTu_RAI.Environments.HammeringEnvironment import HammeringEnvironment

class GenerateEnvironment: 
    def __init__(self, cfg):
        self.cfg = cfg
        self.C = ry.Config()
        self.C.addFile('./rai-robotModels/scenarios/pandaSingle.g')
        
        self.camera = CameraRAI(self.cfg, self.C)

        # 27.01: Added for updated Heuristic
        self.heuristic = Heuristic(C= self.C, cfg = self.cfg)

    def generate_environment(self):
        if self.cfg["tool-type"] == 'mesh': 
            self.tools = GenerateTools(self.cfg, self.C)
        elif self.cfg["tool-type"] == 'primitive':
            self.tools = GeneratePrimitiveTools(self.cfg, self.C)
        elif self.cfg["tool-type"] == 'realistic':
            self.tools = GenerateRealisticTools(self.cfg, self.C)
        elif self.cfg["tool-type"] == 'distractor': 
            self.tools = GenerateDistractiveTools(self.cfg, self.C)
        else: 
            raise Exception("The tool-type was not found or is currently not implemented.")   
        
        
        if self.cfg["task"] == "grilling":
            self.env = GrillingEnvironment(self.cfg, self.C)
        elif self.cfg["task"] == "lifting":
            self.env = LiftingEnvironment(self.cfg, self.C)
        elif self.cfg["task"] == "minigolf":
            self.env = MinigolfEnvironment(self.cfg, self.C)
        elif self.cfg["task"] == "hammering":
            self.env = HammeringEnvironment(self.cfg, self.C)
        else:
            raise Exception("The environment was not found or is currently not implemented.")
        
        self.heuristic.setEnvironment(self.env)
        
        self.bounding_boxes = {}
        self.point_clouds_labels = None
        
        # Add tools
        self.tool_objs = []
        # print(self.tools.getTools())
        
        if self.cfg["mode"] == 'train' or self.cfg["mode"] == 'test':
            random_tools = random.sample(self.tools.getTools(), self.cfg["num-tools"])
        elif self.cfg["mode"] == 'selection_test': 
            good_tools = random.sample(self.tools.getPrimaryTools(), 1)
            bad_tools = random.sample(self.tools.getNonOptimalTools(),self.cfg["num-tools"]-1)
            random_tools = good_tools + bad_tools
        else: 
            raise Exception("something is wrong with mode")
        for too in random_tools:
            self.place_tools_realistic(too)
        
        # self.C.view()


        # Add environment
        self.env_objs = self.env.getObjects()
        print("AAA", self.env_objs)
        for eobj in self.env_objs:
            self.place_environment_simple(eobj)


        # self.C.view()
        

        # self.C.view()
        # time.sleep(7.0)

        print("Inside the generate environment function.")
        # Visualize:
        xyz = np.ascontiguousarray(self.point_clouds_labels[:, :3])  # x, y, z
        labels = np.ascontiguousarray(self.point_clouds_labels[:, 3]) # labels



        print(f"xyz: {xyz.shape}, {xyz.dtype}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Assign colors to points based on labels (e.g., colormap)
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(labels / unique_labels.max())[:, :3]  # Normalize labels and get RGB colors
        colors = np.ascontiguousarray(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        # o3d.visualization.draw_geometries([pcd])

        print("At the end of generate environment function.")
        
    def place_environment_simple(self, env_function):
        env_name = env_function()

        trial = 10
        while True: 
            pos, qua, angle = self.select_random_placement(env_name, "environment", rotate = True)
            print(f"Before Update {qua}") 
            
            self.C.getFrame(env_name).setPosition(self.C.getFrame("table").getPosition() + pos)        
            qua = self.env.update_platform_rotation(qua)
            print(f"After Update {qua}")
            center = self.env.getCenter()
            bounding_box = mutils.create_bounding_box(self.env.bounding_box_shape, center, qua)
            bounding_box["rotation_z"] = angle
            print(bounding_box)
            # mutils.draw_bounding_box(self.C, bounding_box, qua, env_name + "_bb")

            collision = self.check_collision_with_existing(bounding_box)
            if collision == False:
                break 
            trial -=1
            if trial <0:
                raise PlacementError("Can not place the object.")
        
        self.env.setTargetPosition()
        self.env.setTargetQuaternion()

    
        self.bounding_boxes[env_name] = bounding_box
        print("env name: ", env_name)
        pc , _  = self.getPointCloud()
        label = self.cfg["label-list-all"].index(env_name)

        new_points = np.setdiff1d(
                pc.view([('', pc.dtype)] * pc.shape[1]),
                self.point_clouds_labels[:, :3].view([('', self.point_clouds_labels.dtype)] * pc.shape[1])
            ).view(pc.dtype).reshape(-1, pc.shape[1])
        label_array = np.full((new_points.shape[0], 1), label)
        pc_l = np.hstack((new_points, label_array))
        self.point_clouds_labels = np.vstack((self.point_clouds_labels, pc_l))
    
    # Currently the rotation is limited to 30 degrees
    def place_tools_realistic(self,obj_function):
        obj_name = obj_function()
        obj_handle = obj_name  + "-base"
        self.tool_objs.append(obj_name)
        # bounding_box_dimension = [0.07, 0.18, 0.05]
        tool_bounding_boxes = {
                "hammer": [0.07, 0.16, 0.05],
                "spatula": [0.07, 0.15, 0.05],
                "L-ruler": [0.15, 0.16, 0.04],
                "screwdriver" : [0.07, 0.15, 0.05],
                "ball" : [0.08, 0.08, 0.08],
                "ring" : [0.10, 0.10, 0.05],
                "thin-stick": [0.02, 0.12, 0.02],
                "book" : [0.10, 0.15, 0.06]
            }
        bounding_box_dimension = tool_bounding_boxes[obj_name]
        trial_count = 0
        while True:
            pos, qua, angle = self.select_random_placement(obj_handle, "tools", rotate = True)
            
            ball_radius = self.C.getFrame(obj_handle).info()["size"][2] / 2
            table_height = self.C.getFrame("table").info()["size"][2]  # full height of table
            min_required_z = table_height / 2 + ball_radius
            if pos[2] < min_required_z:
                pos[2] = min_required_z



            self.C.getFrame(obj_handle).setPosition(self.C.getFrame("table").getPosition() + pos)
            center = self.C.getFrame(obj_handle).getPosition()
            if obj_name == "ring":
                print("obj_name: ", obj_name)
                center = (self.C.getFrame(obj_handle).getPosition() + self.C.getFrame(obj_name + "-head").getPosition())/2
            bounding_box = mutils.create_bounding_box(bounding_box_dimension, center, qua)
            bounding_box["rotation_z"] = angle
            print(bounding_box)
            # mutils.draw_bounding_box(self.C, bounding_box, qua, obj_name + "_bb")

            collision = self.check_collision_with_existing(bounding_box)
            if collision == False:
                break 

            trial_count += 1
            if trial_count >= 10:
                raise PlacementError(f"Placement failed for after 20 attempts")

        self.bounding_boxes[obj_name] = bounding_box

        # In here I need to get the point cloud. 
        pc , _  = self.getPointCloud()
        label = self.cfg["label-list-all"].index(obj_name)
        
        if self.point_clouds_labels is None: 
            label_array = np.full((pc.shape[0], 1), label)
            pc_l = np.hstack((pc, label_array))
            self.point_clouds_labels = pc_l
        else: 
            # new_points = pc - self.point_clouds_labels[:, :3]
            new_points = np.setdiff1d(
                pc.view([('', pc.dtype)] * pc.shape[1]),
                self.point_clouds_labels[:, :3].view([('', self.point_clouds_labels.dtype)] * pc.shape[1])
            ).view(pc.dtype).reshape(-1, pc.shape[1])
            
            # print(f"New point cloud: {pc.shape}")
            # print(f"Old points shape: {self.point_clouds_labels.shape}")
            # print(f"New points shape: {new_points.shape}")

            label_array = np.full((new_points.shape[0], 1), label)
            pc_l = np.hstack((new_points, label_array))

            self.point_clouds_labels = np.vstack((self.point_clouds_labels, pc_l))

        print(f"obj name: {obj_name}, label: {label}")

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcl)
        # o3d.visualization.draw_geometries([pcd])
    
    def set_tool_objs(self, tool_list):
        self.tool_objs = []
        for tool in tool_list: 
            self.tool_objs.append(tool)

    def place_tools_simple(self, obj_function):
        obj_name, obj_info = obj_function()
        self.tool_objs.append(obj_name)
        trial = 10

        while True: 
            # Fix this first line for mesh objects
            pos, qua = self.select_random_placement(obj_info[0], "tools", rotate = False)
            bounding_box = self.tools.create_bounding_box(self.C, obj_name, obj_info)
            
            # mutils.draw_bounding_box(self.C, bounding_box, qua, obj_name + "_bb")

            collision = self.check_collision_with_existing(bounding_box)
            if collision == False:
                break 
            trial -=1
            if trial <0:
                raise PlacementError("Can not place the object.")

        self.bounding_boxes[obj_name] = bounding_box

    def place_tools_simple_mesh(self,obj_function):
        obj_name, mesh = obj_function()
        self.tool_objs.append(obj_name)
        trial = 10
        while True: 
            pos, qua = self.select_random_placement(obj_name, "tools", rotate = False)
            bounding_box = mutils.create_mesh_bounding_box(self.C, obj_name, mesh)
            
            # mutils.draw_bounding_box(self.C, bounding_box, qua, obj_name + "_bb")

            collision = self.check_collision_with_existing(bounding_box)
            if collision == False:
                break 
            trial -=1
            if trial <0:
                raise Exception("Can not place the object.")

        self.bounding_boxes[obj_name] = bounding_box
    
    def getScore(self):
        print("score tool name: ", self.selected_tool)
        tool_new = self.C.getFrame(self.selected_tool).getPosition()
        obj_new = self.C.getFrame(self.env.getTargetName()).getPosition()
        return self.env.get_score(self.tool_pos, tool_new, obj_new)
    

    def select_random_placement(self, obj, typ, rotate = True):
        if rotate == True:
            # qua = self.generate_random_quaternion()
            qua, angle = self.generate_random_limited_quaternion(typ)
            self.C.getFrame(obj).setQuaternion(qua)
        else: 
            qua = None
            angle =  0

        # pos = self.generate_random_point(typ)
        pos =self.generate_random_point_smarter(typ)


        return pos, qua, angle


    def save_environment_path(self, data_path):
        g_string = self.C.write()  # returns the .g content as string
        with open(data_path, 'w') as f:
            f.write(g_string)
        print(f"Environment saved to {data_path}")
        
    def save_environment(self, data_number):
        data_path = os.path.join(self.cfg['dataset-save-path'], "data_" + str(data_number) + "_env.g")
        g_string = self.C.write()  # returns the .g content as string
        with open(data_path, 'w') as f:
            f.write(g_string)
        print(f"Environment saved to {data_path}")
    
    def save_clean_environment(self, data_number):
        data_path = os.path.join(self.cfg['dataset-save-path'], "data_" + str(data_number) + "_env.g")
        # Remove empty point cloud frames
        for f in list(self.C.frames()):
            try:
                shape_type = f.info().get('shape', None)  # may return None
                if shape_type == 'pointCloud':
                    print(f"Removing empty point cloud frame: {f.name}")
                    self.C.delFrame(f.name)
            except:
                # If .info() not available, fallback to name-based cleanup
                if "pointCloud" in f.name:
                    print(f"Removing point cloud frame by name: {f.name}")
                    self.C.delFrame(f.name)

        # Save the cleaned environment
        g_string = self.C.write()
        with open(data_path, 'w') as f:
            f.write(g_string)
        print(f"Saved cleaned environment to {data_path}")

    def saveSample(self, data_number, pc, selected_tool, waypoints, score):
        data_path = os.path.join(self.cfg['dataset-save-path'], "data_" + str(data_number) + "_pc.npy")
        
        if os.path.exists(data_path):
            raise Exception("This data is already exist in the dataset.")
        print(data_path)
        np.save(data_path, pc)

        
        new_sample = {
            'path': data_number,
            "task" : self.cfg["task"] + '-platform',
            "selected_tool": selected_tool,
            "tool_waypoint": json.dumps(waypoints[0].tolist()),  # Convert numpy array to list and then JSON string
            "initial_waypoint": json.dumps(waypoints[1].tolist()), 
            "goal_waypoint": json.dumps(waypoints[2].tolist()), 
            "score": score["score"], 
            "grasp_score": score["grasp_score"],
            "tak_score" : score["task_score"],
        }

        with open(os.path.join(self.cfg['dataset-save-path'], 'dataset_info.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_sample.keys())
            writer.writerow(new_sample)

    def getPointCloud(self):
        pcl, rgb, _ = self.camera.getPointCloud()
        pcl, rgb  = self.camera.cleanPointClouds(pcl, rgb)

        return pcl, rgb

        


    # This function selects a random tool in data collection part
    def select_random_tool(self):
        selected_tool = random.choice(self.tool_objs)
        # self.tool_pos = self.C.getFrame(selected_tool+'-base').getPosition()
        # self.selected_tool = selected_tool + '-base'
        
        # Find the label of selected tool
        tool_idx = self.cfg["label-list-all-v2"].index(selected_tool)

        return selected_tool, tool_idx
    
    # 27.01: This function is for previous Heuristic, without adding antipodal grasp candidate generation
    def set_waypoints_old(self, selected_tool = None, waypoint = None):
        tool_pose = self.tools.set_grasping_waypoint(tool_name = selected_tool, waypoint=waypoint)
        initial_pose, goal_pose = self.env.get_training_waypoints()

        return [tool_pose, initial_pose, goal_pose]
    
    # 10.02: This func
    def set_waypoints(self, pcl_dict = None, waypoint = None, selected_tool = None, other_tools = None):
        if pcl_dict is None and waypoint is None:
            raise Exception("To be able to set the waypoints, either pcl_dict or waypoint should be given.")
        if pcl_dict is not None: 
            waypoints = self.heuristic.create_heuristic_waypoints(pcl_dict, selected_tool, other_tools)
        if waypoint is not None: 
            self.add_model_waypoints(waypoint)
        return waypoints
    
    def correct_quaternion_to_upward_z(self, predicted_quat):
        """
        Rotates the predicted quaternion so that its local Z-axis aligns with the world Z-axis.

        Args:
            predicted_quat (np.ndarray): Quaternion as [x, y, z, w]

        Returns:
            np.ndarray: Corrected quaternion as [x, y, z, w]
        """
        # Convert to rotation matrix
        rot = R.from_quat(predicted_quat, scalar_first=True)
        R_mat = rot.as_matrix()

        # Get current Z axis
        current_z = R_mat[:, 2]

        # Desired Z axis
        target_z = np.array([0, 0, 1])

        # Compute axis and angle between current and target
        axis = np.cross(current_z, target_z)
        norm_axis = np.linalg.norm(axis)
        
        if norm_axis < 1e-6:
            # Already aligned or opposite
            if np.dot(current_z, target_z) > 0:
                return predicted_quat  # already aligned
            else:
                # 180-degree rotation around any perpendicular axis
                axis = np.array([1, 0, 0])
                angle = np.pi
        else:
            axis = axis / norm_axis
            angle = np.arccos(np.clip(np.dot(current_z, target_z), -1.0, 1.0))

        # Create correction rotation
        correction = R.from_rotvec(axis * angle)

        # Apply correction
        corrected_rot = correction * rot
        return corrected_rot.as_quat(scalar_first=True)

    def post_process_waypoints(self, tool_pc, task_pc, waypoints):
        threshold = 0.02
        line_tol = 0.01
        tool_waypoint = waypoints["pos"][0]
        dists = np.linalg.norm(tool_pc - tool_waypoint, axis=1)
        min_dist = np.min(dists)
        if min_dist < threshold:
            print(f"**DEBUG** The tool waypoint is inside of the tool.")
        else: 
            print(f"**DEBUG** The tool waypoint is not inside of the tool.")
            print(f"**DEBUG** Starting the post processing...")
            
            idx = np.argmin(dists)
            nearest_pt = tool_pc[idx]
            line_vec = tool_waypoint - nearest_pt
            norm = np.linalg.norm(line_vec)

            line_dir = line_vec / norm
            
            # Step 3: Project all tool points onto the line
            vecs = tool_pc - nearest_pt  # (N, 3)
            proj_lengths = vecs @ line_dir  # (N,)
            proj_points = nearest_pt + np.outer(proj_lengths, line_dir)  # (N, 3)

            # Step 4: Find tool points close to the line
            perp_dists = np.linalg.norm(tool_pc - proj_points, axis=1)
            mask = perp_dists < line_tol
            close_points = tool_pc[mask]

            if len(close_points) >=2:
                updated_tool_waypoint = close_points.mean(axis=0)
                z_min = tool_pc[:, 2].min()
                z_max = tool_pc[:, 2].max()
                if tool_waypoint[2] < z_min or tool_waypoint[2] > z_max:
                    z_avg = (z_min + z_max) / 2.0
                    updated_tool_waypoint[2] = z_avg
                else: 
                    updated_tool_waypoint[2] = waypoints["pos"][0][2]
                waypoints["pos"][0] = updated_tool_waypoint
            
                print(f"**DEBUG** Original tool waypoint = {tool_waypoint}")
                print(f"**DEBUG** Post pro tool waypoint = {updated_tool_waypoint}")


        # correct_tool_quat = self.correct_quaternion_to_upward_z(waypoints['qua'][0])
        # waypoints['qua'][0] = correct_tool_quat
        
        # Correct Minigolf if there is a problem: 
        z_values = task_pc[:, 2]
        # Define bin width (e.g., 0.01 for 1cm)
        bin_width = 0.01

        # Discretize z-values into bins
        binned = np.round(z_values / bin_width) * bin_width

        # Find the mode (most frequent bin)
        unique_bins, counts = np.unique(binned, return_counts=True)
        mode_z = unique_bins[np.argmax(counts)]
        if waypoints["pos"][1][2] < mode_z:
            print(f"**DEBUG** The initial waypoint is inside of the platform fixing it.")
            waypoints["pos"][1][2] = mode_z + 0.05
        else: 
            print(f"**DEBUG** The initial waypoint has no problem")
        return waypoints

    # The waypoints are taken from the model
    def add_model_waypoints(self, waypoints):
        tool_waypoint = waypoints["pos"][0] # + np.array([0.0, 0.0, -0.05])
        print(f"**DEBUG** quaternion: {waypoints['qua'][0]}")
        self.C.addFrame("tool-waypoint")\
            . setShape(ry.ST.marker, [.1])\
            .setPosition(tool_waypoint)\
            .setQuaternion(waypoints["qua"][0])
        self.C.addFrame("initial-waypoint")\
            . setShape(ry.ST.marker, [.1])\
            .setPosition(waypoints["pos"][1])\
            .setQuaternion(waypoints["qua"][1])
        self.C.addFrame("goal-waypoint")\
            . setShape(ry.ST.marker, [.1])\
            .setPosition(waypoints["pos"][2])\
            .setQuaternion(waypoints["qua"][2])
        
    # This function is not completed - the placement is missing
    def place_objects(self, objs, type):
        self.bounding_boxes = {}

        # This for loop might need some modifications
        for obj in objs: 
            if obj in self.cfg["object-designs"]:
                while True: 
                    _, qua = self.select_random_placement(self.cfg["object-designs"][obj][0], type)
                    part_bb = mutils.create_primitive_bb(self.C, self.cfg["object-designs"][obj])
                    
                    if len(part_bb) > 1: 
                        # print("obj: ", obj)
                        obj_bb = mutils.combine_primitive_bb(part_bb)
                    else: 
                        obj_bb = part_bb[0]

                    collision = self.check_collision_with_existing(obj_bb)
                    if collision == False:
                        break                 
            else:
                obj_bb = mutils.create_primitive_bb(self.C, obj)
            
            self.bounding_boxes[obj] = obj_bb
            mutils.draw_bounding_box(self.C, obj_bb, qua, obj + "_bb")
    
    def check_collision_with_existing(self, new_box):
        # print(self.bounding_boxes.items())
        # raise Exception
        for _, existing_box in self.bounding_boxes.items():
            if self.is_colliding_obb(new_box, existing_box):
            # if self.is_colliding_obb_relax(new_box, existing_box, tolerance=0.1):
                # print("Yes collide")
                return True
        # print("No collide")
        return False
    

    def generate_random_point(self, typ):
        if typ == "tools": # "environment":
            area = self.cfg["area-middle"]
        elif typ == "environment": # "tools": 
            q = random.uniform(0,1)
            area = self.cfg["area-negative"] if q < 0.5 else self.cfg["area-positive"]
            
        else: 
            raise Exception("Cannot find the type of the area")
        # print(area)
        x = random.uniform(area['min'][0], area['max'][0])
        y = random.uniform(area['min'][1], area['max'][1])
        z = random.uniform(area['min'][2], area['max'][2])

        return [x, y, z]


    # Smarter random placement: 
    def generate_random_point_smarter(self, typ, max_attempts=30, buffer=0.01):
        if typ == "tools":
            area = self.cfg["area-middle"]
        elif typ == "environment":
            q = random.uniform(0, 1)
            area = self.cfg["area-negative"] if q < 0.5 else self.cfg["area-positive"]
            # area = self.cfg["area-positive"] # For Debug
        else:
            raise Exception("Unknown area type!")

        attempt = 0
        while attempt < max_attempts:
            x = random.uniform(area['min'][0], area['max'][0])
            y = random.uniform(area['min'][1], area['max'][1])
            z = random.uniform(area['min'][2], area['max'][2])
            new_pos = [x, y, z]

            if not self.collides_with_existing_bounding_boxes(new_pos, buffer=buffer):
                return new_pos

            attempt += 1

        raise Exception("Could not find a collision-free placement after many tries.")

    def collides_with_existing_bounding_boxes(self, pos, buffer=0.01):
        px, py, pz = pos

        for bb in self.bounding_boxes.values():
            min_pt = bb['min'] - buffer
            max_pt = bb['max'] + buffer
            if (min_pt[0] <= px <= max_pt[0] and
                min_pt[1] <= py <= max_pt[1] and
                min_pt[2] <= pz <= max_pt[2]):
                return True
        return False


    
    def generate_random_quaternion(self):
        theta = np.random.uniform(0, 2 * np.pi)
        
        w = np.cos(theta / 2)  
        x = 0 
        y = 0  
        z = np.sin(theta / 2)  
        
        return np.array([w, x, y, z])
    
    def generate_random_limited_quaternion(self, typ):
        if typ == "tools": # "environment":
            allowed_ranges = [(-90, 90)]
        elif typ == "environment": # "tools": 
            allowed_ranges = [(-30, 30)]
        else: 
            raise Exception("Cannot find the type of the object")
        # allowed_ranges = [(-90, 90)] # [(-30, 30), (150, 210)]
        selected_range = allowed_ranges[np.random.choice(len(allowed_ranges))]
        random_angle = np.random.uniform(selected_range[0], selected_range[1])
        random_angle_rad = np.deg2rad(random_angle)

        rotation_z = R.from_euler('z', random_angle_rad)
        quaternion = rotation_z.as_quat()

        return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]], random_angle
    
    def is_colliding(self, box1, box2):
        if box1['max'][0] < box2['min'][0] or box1['min'][0] > box2['max'][0]:
            return False
        if box1['max'][1] < box2['min'][1] or box1['min'][1] > box2['max'][1]:
            return False
        if box1['max'][2] < box2['min'][2] or box1['min'][2] > box2['max'][2]:
            return False
        return True
    
    #### For rotated bounding boxes. I am not sure :(

    def is_colliding_obb(self, box1, box2):
        corners1 = self.get_corners_from_aabb(box1)
        corners2 = self.get_corners_from_aabb(box2)

        transformed_corners1 = self.rotate_and_translate_corners(corners1, box1['quaternion'], box1['center'])
        transformed_corners2 = self.rotate_and_translate_corners(corners2, box2['quaternion'], box2['center'])



        axes1 = np.array([
            [1, 0, 0],  # Local X-axis
            [0, 1, 0],  # Local Y-axis
            [0, 0, 1]   # Local Z-axis
        ])
        rotation1 = R.from_quat([box1['quaternion'][1], box1['quaternion'][2], box1['quaternion'][3], box1['quaternion'][0]])
        axes1 = rotation1.apply(axes1)

        axes2 = np.array([
            [1, 0, 0],  # Local X-axis
            [0, 1, 0],  # Local Y-axis
            [0, 0, 1]   # Local Z-axis
        ])
        rotation2 = R.from_quat([box2['quaternion'][1], box2['quaternion'][2], box2['quaternion'][3], box2['quaternion'][0]])
        axes2 = rotation2.apply(axes2)

        # Separating axes: axes1, axes2, and cross-products of all pairs of axes
        axes = np.vstack([axes1, axes2, np.cross(axes1[:, None], axes2).reshape(-1, 3)])
        axes = axes[np.linalg.norm(axes, axis=1) > 1e-6]

        # Check for overlap along all axes
        for axis in axes:
            if np.linalg.norm(axis) < 1e-6:
                continue
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            projection1 = np.dot(transformed_corners1, axis)
            projection2 = np.dot(transformed_corners2, axis)

            if np.max(projection1) < np.min(projection2) or np.max(projection2) < np.min(projection1):
                return False  # Separating axis found, no collision

        return True
    
    def is_colliding_obb_relax(self, box1, box2, tolerance=0.005):
        corners1 = self.get_corners_from_aabb(box1)
        corners2 = self.get_corners_from_aabb(box2)

        transformed_corners1 = self.rotate_and_translate_corners(corners1, box1['quaternion'], box1['center'])
        transformed_corners2 = self.rotate_and_translate_corners(corners2, box2['quaternion'], box2['center'])

        axes1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        axes1 = R.from_quat([box1['quaternion'][1], box1['quaternion'][2], box1['quaternion'][3], box1['quaternion'][0]]).apply(axes1)

        axes2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        axes2 = R.from_quat([box2['quaternion'][1], box2['quaternion'][2], box2['quaternion'][3], box2['quaternion'][0]]).apply(axes2)

        axes = np.vstack([axes1, axes2, np.cross(axes1[:, None], axes2).reshape(-1, 3)])
        axes = axes[np.linalg.norm(axes, axis=1) > 1e-6]

        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            projection1 = np.dot(transformed_corners1, axis)
            projection2 = np.dot(transformed_corners2, axis)

            # 💡 Apply soft margin
            if np.max(projection1) < np.min(projection2) - tolerance or np.max(projection2) < np.min(projection1) - tolerance:
                return False  # Separating axis found

        return True  # No separating axis → collision


    def get_corners_from_aabb(self,bounding_box):
        min_point = bounding_box["min"]
        max_point = bounding_box["max"]

        return np.array([
            [min_point[0], min_point[1], min_point[2]],  # Bottom-back-left
            [min_point[0], min_point[1], max_point[2]],  # Bottom-back-right
            [min_point[0], max_point[1], min_point[2]],  # Bottom-front-left
            [min_point[0], max_point[1], max_point[2]],  # Bottom-front-right
            [max_point[0], min_point[1], min_point[2]],  # Top-back-left
            [max_point[0], min_point[1], max_point[2]],  # Top-back-right
            [max_point[0], max_point[1], min_point[2]],  # Top-front-left
            [max_point[0], max_point[1], max_point[2]],  # Top-front-right
        ])
    
    def rotate_and_translate_corners(self, corners, quaternion, center):
        quaternion = quaternion / np.linalg.norm(quaternion)
        corners_local = corners - center
        

        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # Convert [w, x, y, z] to [x, y, z, w]

        rotated_corners = rotation.apply(corners_local)
        transformed_corners = rotated_corners  + center
        # print("transformed corners: ", transformed_corners)
        return transformed_corners
    


class PlacementError(Exception):
    """Raised when a collision free placement cannot be found for current localization by the limited number of trials."""
    pass


