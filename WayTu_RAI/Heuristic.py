import random
import numpy as np 
import robotic as ry
import open3d as o3d
from scipy.spatial import KDTree 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import scipy.spatial.transform as tf
from scipy.spatial.transform import Rotation as R

import WayTu_RAI.model_utils as mutils

class Heuristic:
    def __init__(self, C, cfg):
        self.C = C
        self.cfg = cfg
        self.environment = None

        self.max_distance = 0.08
        self.min_distance = 0.005
        self.angle_threshold = - 0.95
    
    def setEnvironment(self, environment):
        self.environment = environment
    
    def create_heuristic_waypoints(self, pcl_dict, selected_tool, other_tools):
        # pcl_dict keys: tool_pc, environment_pc
        tool_pc = pcl_dict["tool_pc"]
        print(f"tool_pc: {tool_pc.shape}")

        # Calculate the normals 
        k = 20
        attempt, tw_bool = 0, False
        updated_tool_pc = tool_pc
        while attempt < 1:
            print(tw_bool)
            if tw_bool:
                print("I found the points") 
                break
            tool_pcd = o3d.geometry.PointCloud()
            tool_pcd.points = o3d.utility.Vector3dVector(updated_tool_pc)
        
            tool_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k))
            tool_pcd.orient_normals_consistent_tangent_plane(k=k)

            tw_bool, self.tool_waypoint = self.calculate_grasp_waypoint(tool_pcd)
            updated_tool_pc = mutils.upsample_point_clouds(updated_tool_pc)
            updated_tool_pc = mutils.farthest_point_sampling(updated_tool_pc, k=tool_pc.shape[0]) # Added for decreasing calculation time -10.04

            attempt += 1

        if not tw_bool:
            print("The antipodal grasping algorithm couldn't found good grasping point. Center is selected")
            tool_waypoint_position = self.C.getFrame(selected_tool+"-base").getPosition()
            tool_waypoint_quaternion = self.C.getFrame(selected_tool+"-base").getQuaternion()
            self.add_waypoint("tool-waypoint", tool_waypoint_position, tool_waypoint_quaternion)
            self.tool_waypoint = np.concatenate([tool_waypoint_position, tool_waypoint_quaternion])

        
        # Reparent the tool for grasping:
        selected_part = mutils.find_grasping_part(self.C, selected_tool, self.tool_waypoint)
        
        mutils.reparent_tool(self.C, selected_tool, selected_part, other_tools)
        print(f"Selected Part: {selected_part}")
        try: 
            contact_point = self.find_contact_point(tool_pcd, self.tool_waypoint[:3])
        except ValueError:
            dists = np.linalg.norm(tool_pc - self.tool_waypoint[:3], axis=1)
            contact_point = tool_pc[np.argmax(dists)]
            
        # self.calculate_initial_waypoint(contact_point, pcl_dict["environment_pc"])
        # self.calculate_initial_waypoint_debug(contact_point)
        self.calculate_initial_waypoint_goal(contact_point)
        self.calculate_goal_waypoint()
        # raise Exception
    
        # target = self.C.getFrame("target")
        # target_center = target.getPosition()


        # # (1) Find the grasped region because it is inaccessible for interaction
        # grasped_region = ...
        waypoints = [
            np.concatenate((self.C.getFrame("tool-waypoint").getPosition(),self.C.getFrame("tool-waypoint").getQuaternion())),
            np.concatenate((self.C.getFrame("initial-waypoint").getPosition(),self.C.getFrame("initial-waypoint").getQuaternion())),
            np.concatenate((self.C.getFrame("goal-waypoint").getPosition(),self.C.getFrame("goal-waypoint").getQuaternion())),

        ]
        return waypoints
    
    def draw_points(self, name, position, color = [1,0,0, 0.5]):
        self.C.addFrame(name) \
            .setPosition(position) \
            .setShape(ry.ST.ssBox, size=[.03,.03,.03,.005]) \
            .setColor(color)
    
    def add_waypoint(self, name, position, quaternion):
        self.C.addFrame(name)\
            . setShape(ry.ST.marker, [.1])\
            .setPosition(position)\
            .setQuaternion(quaternion)
    
    # def get_goal(self):
    #     if self.cfg["task"] == "lifting":
    #         height_offset = np.random.uniform(0.05, 0.15)
    #         return np.array([0, 0, height_offset])
    #     elif self.cfg["task"] == "minigolf":
    #         hole_center = (self.C.getFrame("left-area").getPosition() + self.C.getFrame("right-area").getPosition())/2
    #         hole_center[2] = self.getFrame("minigolf-obj").getPosition()[2]

    def calculate_grasp_waypoint(self, tool_pcd):
        # o3d.visualization.draw_geometries([tool_pcd], point_show_normal=True)
        # The problem, when the selected tool is spatula, it cannot find any pairs. We need to update the structure. 
        grasp_pairs = self.antipodal_grasp_generation(tool_pcd)
        
        if grasp_pairs.shape[0] == 0:
            return False, None
        
        weights = grasp_pairs[:, 2].astype(float)
        grasp_antipodal = random.choices(grasp_pairs.tolist(), weights=weights, k=1)[0]

        tool_pc = tool_pcd.points
        # Visualize and print the antipodal points
        print(f"The points proposed for antipodal grasp generation: {grasp_antipodal}") 
        print(f"Point1: {tool_pc[int(grasp_antipodal[0])]}")
        print(f"Point2: {tool_pc[int(grasp_antipodal[1])]}")

        # self.draw_points("point1",tool_pc[int(grasp_antipodal[0])])
        # self.draw_points("point2",tool_pc[int(grasp_antipodal[1])])

        # Calculate the tool/grasping waypoint
        point1_idx, point2_idx = int(grasp_antipodal[0]), int(grasp_antipodal[1])
        p1, p2 = tool_pcd.points[point1_idx], tool_pcd.points[point2_idx]
        n1, n2 = tool_pcd.normals[point1_idx], tool_pcd.normals[point2_idx]
        waypoint_position = (tool_pcd.points[point1_idx] + tool_pcd.points[point2_idx]) / 2
        
        print(f"Waypoint Position: {waypoint_position}")
        if waypoint_position[2] < 0.66:
            waypoint_position[2] == np.mean(tool_pc[2])

        grasp_x = n1 / np.linalg.norm(n1)  

        # Step 2: Manually force Z-axis to be perpendicular to XY plane
        grasp_z = np.array([0, 0, 1])  # This ensures Z always points upwards correctly

        # Step 3: Compute Y-axis using cross product to ensure a right-handed frame
        grasp_y = np.cross(grasp_z, grasp_x)
        grasp_y /= np.linalg.norm(grasp_y)  # Normalize

        # grasp_x = (tool_pcd.points[point2_idx] - tool_pcd.points[point1_idx]) / np.linalg.norm(tool_pcd.points[point2_idx] - tool_pcd.points[point1_idx])
        # grasp_z = (tool_pcd.normals[point1_idx] + tool_pcd.normals[point2_idx]) / np.linalg.norm(tool_pcd.normals[point1_idx] - tool_pcd.normals[point2_idx])
        # grasp_y = np.cross(grasp_z, grasp_x)
        # grasp_y /= np.linalg.norm(grasp_y)

        # grasp_x = np.cross(grasp_y, grasp_z)

        R = np.column_stack((grasp_x, grasp_y, grasp_z))
        waypoint_quaternion = mutils.rotation_matrix_to_quaternion(R)
        self.add_waypoint("tool-waypoint", waypoint_position, waypoint_quaternion)

        self.C.view()
        return True, np.concatenate([waypoint_position, waypoint_quaternion])

    
    def antipodal_grasp_generation(self, tool_pcd):
        # mutils.visualize_pc_with_labels(tool_pcd.points)
        points = np.asarray(tool_pcd.points)
        normals = np.asarray(tool_pcd.normals)
        
        kdtree = KDTree(points)

        # For debugging: 
        rejected_by_distance = 0
        rejected_by_angle = 0
        accepted_pairs = 0

        grasp_pairs = []
        for i in range(points.shape[0]):
            neighbors = kdtree.query_ball_point(points[i], self.max_distance)

            for j in neighbors:
                if i == j: 
                    continue

                distance = np.linalg.norm(points[i] - points[j])
                if distance < self.min_distance:
                    rejected_by_distance += 1
                    continue

                normal_dot = np.dot(normals[i], normals[j])
                if normal_dot <= self.angle_threshold:
                    grasp_vector = points[j] - points[i]
                    grasp_vector /= np.linalg.norm(grasp_vector)

                    if np.dot(grasp_vector, normals[i]) < self.angle_threshold and np.dot(grasp_vector, normals[j]) > self.angle_threshold:
                        score = -normal_dot / distance
                        grasp_pairs.append((i, j, score))
                        accepted_pairs += 1
                    else:
                        rejected_by_angle += 1
                else:
                    rejected_by_angle += 1

        grasp_pairs = np.array(sorted(grasp_pairs, key=lambda x: x[2], reverse=True))
        
        print(f"Accepted Grasp Pairs: {accepted_pairs}")
        print(f"Rejected by Distance: {rejected_by_distance}")
        print(f"Rejected by Angle: {rejected_by_angle}")
        
        return grasp_pairs

    def find_contact_point(self, tool_pcd, grasp_position):
        tool_pc = np.asarray(tool_pcd.points)  
        tool_normals = np.asarray(tool_pcd.normals)

        radius = 0.1
        distances = np.linalg.norm(tool_pc - grasp_position, axis=1)
        grasping_area_indices = np.where(distances < radius)[0]

        grasping_area_mask = np.ones(len(tool_pc), dtype=bool)
        grasping_area_mask[grasping_area_indices] = False 
        candidate_contact_points = tool_pc[grasping_area_mask]
        candidate_normals = tool_normals[grasping_area_mask]

        # Cluster points by using their normals
        dbscan = DBSCAN(eps=0.1, min_samples=10).fit(candidate_normals)
        labels = dbscan.labels_
        unique_labels = np.unique(labels[labels >= 0])
        surface_indices = [np.where(labels == label)[0] for label in unique_labels]
        
        selected_surface = np.random.choice(len(surface_indices)) 
        random_idx = np.random.choice(surface_indices[selected_surface])

        contact_point = candidate_contact_points[random_idx]

        self.draw_points("contact_point", contact_point)
        self.C.view()

        # raise Exception

        return contact_point
    

    
    def calculate_initial_waypoint_goal(self, tool_contact_point, debug = False):
        # Get target contact point
        target_name = self.cfg["task"] + "-obj"
        target_contact = self.C.getFrame(target_name).getPosition()

        # Get grasp point (from antipodal grasp logic)
        grasp_point = self.tool_waypoint[:3]

        # Compute tool vector and length
        tool_vector = tool_contact_point - grasp_point
        tool_length = np.linalg.norm(tool_vector)
        
        # Get general movement direction (task-dependent, but geometry-driven)
        movement_dir = self.environment.get_movement_direction()  # should return a unit vector
        if np.linalg.norm(movement_dir) < 1e-6:
            raise ValueError("Invalid movement direction.")
        
        # Compute initial waypoint position (gripper pose)
        buffer = 0.015  # safety distance
        initial_position = target_contact - (tool_length + buffer) * movement_dir

        # Compute rotation to align tool with movement direction
        v1 = tool_contact_point - grasp_point  # original grasp axis
        v2 = movement_dir

        rotation_axis = np.cross(v1, v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        if np.linalg.norm(rotation_axis) < 1e-6:
            rot = R.identity()
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rot = R.from_rotvec(angle * rotation_axis)
        
        # Rotate the original tool waypoint orientation
        original_quat = self.tool_waypoint[3:]
        original_rot = R.from_quat(original_quat, scalar_first=True)
        new_rot = rot * original_rot
        initial_quat = new_rot.as_quat(scalar_first=True)

        # Step 8: Visual debug
        self.draw_points("target_contact", target_contact, color=[1.0, 0.0, 0.0, 0.5])
        self.draw_points("tool_contact", tool_contact_point, color=[0.0, 1.0, 0.0, 0.5])
        self.draw_points("initial_pos", initial_position, color=[0.0, 0.0, 1.0, 0.5])

        # Step 9: Add the initial waypoint
        self.add_waypoint("initial-waypoint", initial_position, initial_quat)
        self.C.view()

        # raise Exception



    def calculate_initial_waypoint_debug(self, tool_contact_point):
        # target_contact = self.C.getFrame("minigolf-obj").getPosition()
        # # target_contact = self.C.getFrame("lifting-obj").getPosition()
        # # target_contact = self.C.getFrame("hammering-obj").getPosition()
        
        # Get target contact point for each object
        target_contact = self.C.getFrame(self.cfg["task"] + "-obj").getPosition()
        grasp_point = self.tool_waypoint[:3]    # Position of grasping point    

        v1 = tool_contact_point - grasp_point   # current direction from grasp to contact
        v2 = target_contact - grasp_point       # desired direction from grasp to wall

        # Step 3: Normalize both vectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        rotation_axis = np.cross(v1, v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        
        if np.linalg.norm(rotation_axis) < 1e-6:
            rot = R.identity()
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rot = R.from_rotvec(angle * rotation_axis)

        # Rotate the tool frame
        original_quat = self.tool_waypoint[3:] 
        original_rot = R.from_quat(original_quat, scalar_first=True)
        new_rot = rot * original_rot   
        new_quat = new_rot.as_quat(scalar_first=True)

        rotated_contact = rot.apply(tool_contact_point - grasp_point) + grasp_point
        offset = target_contact - rotated_contact
        shifted_position = grasp_point + offset

        backoff_distance = np.random.uniform(0.01, 0.04)
        direction = rotated_contact - target_contact
        direction /= np.linalg.norm(direction)

        final_position = shifted_position + backoff_distance * direction

        # 6. Visual debug
        self.draw_points("target_contact", target_contact)
        self.draw_points("rotated_contact", rotated_contact, color=[0.0, 0.5, 1.0, 0.5])
        self.add_waypoint("initial-waypoint", final_position, new_quat)
        self.C.view()

        raise Exception

    def calculate_initial_waypoint(self, tool_contact_point, env_pc):
        target_name = self.cfg['task'] + '-obj'
        target_obj = self.C.getFrame(target_name)
        target_size = target_obj.info()['size'][:3]
        target_position = target_obj.getPosition()

        # Get target bounding box limits
        x_min, x_max = target_position[0] - target_size[0] / 2, target_position[0] + target_size[0] / 2
        y_min, y_max = target_position[1] - target_size[1] / 2, target_position[1] + target_size[1] / 2
        z_min, z_max = target_position[2] - target_size[2] / 2, target_position[2] + target_size[2] / 2

        # Filter points inside the target bounding box
        mask = (
            (env_pc[:, 0] >= x_min) & (env_pc[:, 0] <= x_max) &
            (env_pc[:, 1] >= y_min) & (env_pc[:, 1] <= y_max) &
            (env_pc[:, 2] >= z_min) & (env_pc[:, 2] <= z_max)
        )
        target_points = env_pc[mask]

        # Ensure we have valid points
        if len(target_points) == 0:
            raise ValueError("No valid target points found!")

        # Compute target center
        target_center = np.mean(target_points, axis=0)

        # Minigolf-specific adjustments
        if self.cfg["task"] == "minigolf":
            end_area_position = self.C.getFrame("end-area").getPosition()
            hole_direction = end_area_position - target_position
            hole_direction /= np.linalg.norm(hole_direction)  # Normalize direction

            # Remove points that are too close to the hole and below the target
            dot_products = np.dot(target_points - target_position, hole_direction)
            filtered_points = target_points[(dot_products < 0) & (target_points[:, 2] > target_position[2])]

            if len(filtered_points) > 0:
                target_points = filtered_points  # Keep only valid points

        # Randomized selection of target contact point
        randomness = 0.08
        center_distances = np.linalg.norm(target_points - target_center, axis=1)
        center_weights = np.exp(-center_distances / randomness)
        selected_idx = np.random.choice(len(target_points), p=center_weights / np.sum(center_weights))

        target_contact = target_points[selected_idx]

        # For debug: 
        target_contact = self.C.getFrame("hammering-obj").getPosition()

        # Draw debug visualization
        self.draw_points("target_contact", target_contact)
        self.C.view()

        # raise Exception

        # Compute normal using nearest neighbors
        distances = np.linalg.norm(target_points - target_contact, axis=1)
        nearest_indices = np.argsort(distances)[:10]
        nearest_neighbors = target_points[nearest_indices]

        mean = np.mean(nearest_neighbors, axis=0)
        cov_matrix = np.cov(nearest_neighbors - mean, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        target_normal = eigenvectors[:, 0]
        target_normal /= np.linalg.norm(target_normal)  # Normalize

        # Ensure normal points upwards
        if target_normal[2] < 0:
            target_normal = -target_normal  # Flip normal if necessary


        if self.cfg["task"] == "hammering":
            wall_position = self.C.getFrame("hammering-platform").getPosition()
            vector_to_wall = wall_position - target_contact

            # Normalize both vectors
            vector_to_wall /= np.linalg.norm(vector_to_wall)
            target_normal /= np.linalg.norm(target_normal)

            dot_product = np.dot(vector_to_wall, target_normal)

            # If the normal points into the wall, flip it
            if dot_product > 0:
                target_normal = -target_normal
            
            # target_contact = target_contact - 0.02 * target_normal

        # Compute gripper position
        # grasp_vector = contact_point - self.tool_waypoint[:3]
        # grasp_length = np.linalg.norm(grasp_vector)

        # if self.cfg["task"] == "lifting" or self.cfg["task"] == "hammering":
        #     gripper_position = target_contact - grasp_length * target_normal
        # elif self.cfg["task"] == "minigolf":
        #     # Move waypoint slightly back along the centerline from main-area to end-area
        #     main_area_center = self.C.getFrame("main-area").getPosition()
        #     end_area_center = self.C.getFrame("end-area").getPosition()
        #     centerline_direction = end_area_center - main_area_center
        #     centerline_direction /= np.linalg.norm(centerline_direction)  # Normalize

        #     waypoint_offset = -0.05  # Tune this value if needed
        #     gripper_position = target_contact + waypoint_offset * centerline_direction



        # --- More reliable gripper position logic ---
        tool_grasp_point= self.tool_waypoint[:3]
        grasp_vector = tool_contact_point - tool_grasp_point
        grasp_vector /= np.linalg.norm(grasp_vector)  # Normalize

        # Use grasp direction to position gripper
        grasp_length = np.linalg.norm(tool_contact_point - tool_grasp_point)
        safe_offset = 0.02  # Smaller offset, tune if needed

        gripper_position = target_contact - (grasp_length + safe_offset) * grasp_vector

        # Compute rotation to align tool
        v_initial = grasp_vector / np.linalg.norm(grasp_vector)
        v_target = target_normal / np.linalg.norm(target_normal)

        rotation_axis = np.cross(v_initial, v_target)
        if np.linalg.norm(rotation_axis) < 1e-6:  # Already aligned
            quaternion = self.tool_waypoint[3:]
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)
            dot_product = np.clip(np.dot(v_initial, v_target), -1.0, 1.0)
            rotation_angle = np.arccos(dot_product)

            q_align = R.from_rotvec(rotation_angle * rotation_axis).as_quat()
            new_gripper_quaternion = R.from_quat(q_align) * R.from_quat(self.tool_waypoint[3:], scalar_first=True)

            quaternion = new_gripper_quaternion.as_quat(scalar_first=True)

        # Add waypoint
        self.add_waypoint("initial-waypoint", gripper_position, quaternion)
        self.C.view()

        # Debugging output
        debug_info = {
            "Task": self.cfg["task"],
            "Target Position": target_position,
            "Target Contact": target_contact,
            "Target Normal": target_normal,
            "Grasp Length": grasp_length,
            "Computed Gripper Position": gripper_position,
            "Main Area Center": main_area_center if self.cfg["task"] == "minigolf" else "N/A",
            "End Area Center": end_area_center if self.cfg["task"] == "minigolf" else "N/A",
            "Centerline Direction": centerline_direction if self.cfg["task"] == "minigolf" else "N/A",
            "Waypoint Offset": waypoint_offset if self.cfg["task"] == "minigolf" else "N/A",
            "Rotation Axis": rotation_axis,
            "Rotation Axis Norm": np.linalg.norm(rotation_axis),
            "Dot Product": dot_product,
            "Rotation Angle (radians)": rotation_angle
        }
        
        for key, value in debug_info.items():
            print(f"{key}: {value}")
            
    # def calculate_goal_waypoint(self):
    #     initial_waypoint = self.C.getFrame("initial-waypoint")
    #     initial_position = initial_waypoint.getPosition()
    #     initial_quaternion = initial_waypoint.getQuaternion()

    #     goal_position = initial_position + self.get_goal()
    #     goal_quaternion = initial_quaternion

    #     self.add_waypoint("goal-waypoint", goal_position, goal_quaternion)
    #     self.C.view()

    def calculate_goal_waypoint(self):
        goal_position, goal_quaternion = self.environment.calculate_goal_waypoint()
        self.add_waypoint("goal-waypoint", goal_position, goal_quaternion)
        self.C.view()