import os
import re
import robotic as ry
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def patch_all_in_place(g_folder):
    """
    For each .g in g_folder, rewrite any absolute prefix up to 'rai-robotModels'
    to point to your local rai-robotModels directory.
    """
    g_folder = os.path.abspath(g_folder)
    # The correct prefix: parent_of_g_folder + "/rai-robotModels"
    correct_prefix = os.path.join(os.path.dirname(g_folder), "rai-robotModels")

    for fn in os.listdir(g_folder):
        if not fn.lower().endswith(".g"):
            continue
        path = os.path.join(g_folder, fn)
        text = open(path, "r").read()

        # Find the old absolute prefix ending in '.../rai-robotModels'
        match = re.search(r"(/[^'\"]*?/rai-robotModels)", text)
        if match:
            old_prefix = match.group(1)
            if old_prefix != correct_prefix:
                patched = text.replace(old_prefix, correct_prefix)
                with open(path, "w") as f:
                    f.write(patched)
                print(f"Patched {fn}:\n  {old_prefix}\n→ {correct_prefix}\n")
            else:
                print(f"{fn} already correct.")
        else:
            print(f"No rai-robotModels prefix found in {fn}, skipping.")

def fix_g_path(path, msg, input_dir):
    key = "rai-robotModels"
    pattern = r"couldn't change to directory '([^']*)' from '([^']*)'"
    m = re.search(pattern, msg)
    if m:
        old_dir = m.group(1)
        src_dir = m.group(2)
        print(f"old_dir: {old_dir}")
        print(f"src_dir: {src_dir}")
        i = old_dir.find(key)
        if i < 0:
            raise ValueError(f"'{key}' not found in old_dir: {old_dir!r}")
        old_prefix = old_dir[:i] 
        j = src_dir.find(input_dir)
        if j < 0:
            raise ValueError(f"root_dir_name not in src_dir: {src_dir!r}")
        src_prefix = src_dir[:j] 

        path = os.path.join(src_prefix, path)
        print(f"path: {path}")
        print(f"old prefix: {old_prefix}")
        print(f"src_prefix: {src_prefix}")

        # raise Exception
        with open(path, 'r') as f:
            text = f.read()
        # replace
        if old_prefix in text:
            new_text = text.replace(old_prefix, src_prefix)
            # write back
            with open(path, 'w') as f:
                f.write(new_text)
            print(f"Patched prefixes in {path}")
    else:
        print("No match found")

def safe_add_g(env, g_path, input_dir):
    try:
        env.addFile(g_path)
        return
    except Exception as e:
        msg = str(e)
        if "couldn't change to directory" not in msg:
            raise  Exception(msg)# some other 
        else: 
            fix_g_path(path=g_path, msg=msg, input_dir= input_dir)


class ToolBotRAIDatasetAdaptation:
    def __init__(self, input_folder, output_folder, top_n=None):
        self.input_folder  = input_folder
        self.output_folder = output_folder

        if os.path.exists(self.output_folder) == False:
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(os.path.join(output_folder, "inputs"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "outputs"), exist_ok=True)

        if os.path.exists(self.input_folder) == False:    
            raise Exception

        # camera frames in your .g files
        self.camera_names = ["camera1", "camera2", "camera3"]

        # all possible labels in dataset_info.csv
        self.label_list_all = [
          "lifting-platform","minigolf-platform","hammering-platform",
          "hammer","spatula","L-ruler"
        ]
        self.size=(224,224)
        # load metadata
        info_csv = os.path.join(self.input_folder, "dataset_info.csv")
        self.meta = pd.read_csv(info_csv, dtype={"path": int})

        # if top_n is not None:
        #     self.meta = (
        #         self.meta
        #         .sort_values("score", ascending=False)
        #         .head(int(top_n))
        #         .copy()
        #     )
        # def env_path_for_pid(pid: int):
        #     return os.path.join(self.input_folder, f"data_{pid}_env.g")

        # envs = []
        # missing = 0
        # for pid in self.meta["path"].astype(int).tolist():
        #     p = env_path_for_pid(pid)
        #     if os.path.exists(p):
        #         envs.append(p)
        #     else:
        #         missing += 1

        # self.environments = envs

        self.environments = [
            os.path.join(self.input_folder, file_name)
            for file_name in os.listdir(self.input_folder)
            if file_name.lower().endswith('.g')
        ]

    def getEnvironmentPC(self, env_path):
        C = ry.Config()
        safe_add_g(C, env_path, self.input_folder)

        # table & robot cutoffs
        tfrm = C.getFrame("table")
        table_z = tfrm.getPosition()[2] + tfrm.info()["size"][2]/2 + 0.005
        grip_z  = C.getFrame("l_gripper").getPosition()[2] - 0.04

        all_p, all_c, all_d = [],[],[]
        for cam in self.camera_names:
            # pcl, rgb, depth = ry.CameraView(C).setCamera(cam).computeImageAndDepth(C)
            pcl, rgb, depth = self.getPC_from_cam(cam, C)
            
            # pcl = ry.depthImage2PointCloud(depth, ry.CameraView(C).getFxycxy())
            pcl = self.cam_to_world(pcl.reshape(-1,3), C.getFrame(cam))
            
            all_p.append(pcl); all_c.append(rgb.reshape(-1,3)); all_d.append(depth.reshape(-1))
        P = np.vstack(all_p); Cc = np.vstack(all_c); D = np.hstack(all_d)

        mask = (P[:,2]>table_z)&(P[:,2]<grip_z)&(P[:,1]>0)&(P[:,1]<0.6)
        return P[mask], Cc[mask]/256.0, D[mask]
    
    def getPC_from_cam(self, name, C): 
        cam = ry.CameraView(C)
        cam.setCamera(name)
        rgb, depth = cam.computeImageAndDepth(C)
        pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())
        return pcl, rgb, depth

    def cam_to_world(self, point_cloud, cam_frame):
        # Get camera position and rotation matrix
        t = cam_frame.getPosition()  # Camera position in world frame (1x3 array)
        R = cam_frame.getRotationMatrix()  # Camera rotation matrix (3x3)

        # Ensure the point cloud has homogeneous coordinates
        points_camera_frame_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # Construct the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R  # Set the rotation part
        transformation_matrix[:3, 3] = t   # Set the translation part

        # Transform points to world frame
        points_world_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T
        points_world_frame = points_world_frame_homogeneous[:, :3]

        return points_world_frame
    
    def map_labels(self, original_pc, original_labels, new_pc):
        tree = cKDTree(original_pc)
        dists, idx = tree.query(new_pc, k=1)
        max_err = dists.max()
        return original_labels[idx], max_err
    
    def pointcloud_to_heightmaps(self, pcl, rgb, depth,
                                 padding=0.005):
        print(f"**DEBUG** pcl: {pcl.shape} ")
        x,y,z = pcl.T
        H,W   = self.size
        # bounds + padding
        x_min,x_max = x.min()-padding, x.max()+padding
        y_min,y_max = y.min()-padding, y.max()+padding
        # resolution = meters/pixel
        res = max((x_max-x_min)/W, (y_max-y_min)/H)

        depth_map = np.full((H,W), -np.inf, dtype=np.float32)
        rgb_map   = np.zeros((H,W,3),   dtype=np.uint8)

        for (xi, yi, zi), (ri, gi, bi), di in zip(pcl, rgb, depth):
            # Convert world → pixel coords
            col = int((xi - x_min) / res)
            row = H - 1 - int((yi - y_min) / res)  # flip Y so row 0 is top

            if 0 <= row < H and 0 <= col < W:
                # Keep only the highest point in each cell
                if zi > depth_map[row, col]:
                    depth_map[row, col] = zi
                    # Map color to 0–255
                    rgb_map[row, col]   = np.clip(
                        (np.array([ri, gi, bi]) * 255.0), 0, 255
                    ).astype(np.uint8)

        depth_map[depth_map==-np.inf] = 0.0
        # store for waypoint use
        self.bounds = (x_min,x_max,y_min,y_max)
        self.res    = res
        return rgb_map, depth_map
    
    def waypoint_to_heatmap(self, x, y, quat, H, W, K):
        yaw = self.quaternion_to_yaw_scalar_first(quat)
        u   = int((x - self.bounds[0])/self.res + 0.5)
        v   = H - 1 - int((y - self.bounds[2])/self.res + 0.5)
        b   = int((yaw/(2*np.pi))*K) % K
        hm  = np.zeros((K, H, W), dtype=np.float32)
        if 0 <= u < W and 0 <= v < H:
            hm[b, v, u] = 1.0
        return hm
    
    def quaternion_to_yaw(self, quat):
        """
        Converts (qx,qy,qz,qw) into a single yaw ∈ [0,2π).
        """
        qx, qy, qz, qw = quat
        siny = 2 * (qw*qz + qx*qy)
        cosy = 1 - 2 * (qy*qy + qz*qz)
        yaw  = np.arctan2(siny, cosy)
        # normalize into [0,2π)
        return yaw if yaw >= 0 else yaw + 2*np.pi
    
    def quaternion_to_yaw_scalar_first(self, quat_wxyz):
        w,x,y,z = quat_wxyz
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw if yaw >= 0 else yaw + 2*np.pi

    def create_dataset(self):
        data = 0
        for env_path in self.environments:
            print(f"Data Sample: {data}")
            data += 1
            # 1) grab & clean
            pcl, rgb, depth = self.getEnvironmentPC(env_path)
            print(pcl.shape)
            # 2) metadata
            name   = os.path.splitext(os.path.basename(env_path))[0][:-4]
            pid    = int(re.search(r"data_(\d+)", name).group(1))
            print(f"Path id = {pid}")
            sample_info = self.meta[self.meta["path"] == pid]
            task_name = sample_info["task"].iloc[0]
            tool_name = sample_info["selected_tool"].iloc[0]

            task_idx = self.label_list_all.index(task_name)
            tool_idx = self.label_list_all.index(tool_name)
            
            orig   = np.load(os.path.join(self.input_folder, f"{name}_pc.npy"))
            print(f"orig.shape : {orig.shape}")
            o_xyz, o_lbl = orig[:,:3], orig[:,3].astype(int)

            # 3) label-transfer + filter
            new_lbl, error  = self.map_labels(o_xyz, o_lbl, pcl)
            keep_mask = np.logical_or(new_lbl == task_idx,
                                  new_lbl == tool_idx)
            Pf, Rc, Df = pcl[keep_mask], rgb[keep_mask], depth[keep_mask]
            print(f"inside create_dataset: {Pf.shape}")
            # 4) rasterize
            rgb_hm, depth_hm = self.pointcloud_to_heightmaps(Pf, Rc, Df)

            # 5) save heightmaps
            np.save(os.path.join(self.output_folder, "inputs",  f"{name}_rgb.npy"), rgb_hm)
            np.save(os.path.join(self.output_folder, "inputs",  f"{name}_depth.npy"), depth_hm)

            # 6) waypoints→heatmaps
            # D) get waypoints from RAI frames
            C = ry.Config(); # safe_add_g(C, env_path)
            C.addFile(env_path)
            tw = C.getFrame("tool-waypoint")
            iw = C.getFrame("initial-waypoint")
            gw = C.getFrame("goal-waypoint")
            tx,ty,tz = tw.getPosition(); tq = tw.getQuaternion()
            ix,iy,iz = iw.getPosition(); iq = iw.getQuaternion()
            gx,gy,gz = gw.getPosition(); gq = gw.getQuaternion()

            # E) heatmaps: grasp from tool-waypoint; manip = init + goal
            H,W = self.size
            hmg = self.waypoint_to_heatmap(tx, ty, tq, H, W, K=16)
            hmi = self.waypoint_to_heatmap(ix, iy, iq, H, W, K=32)
            hmg2= self.waypoint_to_heatmap(gx, gy, gq, H, W, K=32)
            # combine initial+goal in one manipulation heatmap
            hmm = (hmi + hmg2).clip(0,1)

            # save outputs
            np.save(os.path.join(self.output_folder, "outputs", f"{name}_hmg.npy"), hmg)
            np.save(os.path.join(self.output_folder, "outputs", f"{name}_hmm.npy"), hmm)

if __name__=="__main__":
    # G_FOLDER = "hammering-dataset-heuristic-top1800-all"
    # patch_all_in_place(G_FOLDER)
    builder = ToolBotRAIDatasetAdaptation(
        input_folder="hammering-dataset-heuristic-top1800-all",
        output_folder="toolbot-hammering"
    )
    builder.create_dataset()

# import robotic as ry
# import numpy as np
# import pandas as pd
# import os
# import re
# import open3d as o3d
# from PIL import Image
# import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

# # ry.setRaiPath("/home/ece/git/WayTU-002")
# MAIN_DIR = "/home/ece/git/WayTU-002"   

# class ToolBotRAIDataset:
#     def __init__(self, root_dir, save_dir):
#         self.camera_names = ["camera1", "camera2", "camera3" ]
#         self.label_list_all = [
#             "lifting-platform", "minigolf-platform", "hammering-platform",
#             "hammer", "spatula", "L-ruler"
#         ]

#         self.root_dir = root_dir
#         self.save_dir = save_dir
#         self.environments = [
#             os.path.join(root_dir, file_name)
#             for file_name in os.listdir(root_dir)
#             if file_name.lower().endswith('.g')
#         ]

#         info_path = os.path.join(root_dir, "dataset_info.csv")
#         self.meta = pd.read_csv(info_path)
    
#     def fix_g_path(self, path, msg):
#         key = "rai-robotModels"
#         pattern = r"couldn't change to directory '([^']*)' from '([^']*)'"
#         m = re.search(pattern, msg)
#         if m:
#             old_dir = m.group(1)
#             src_dir = m.group(2)
#             i = old_dir.find(key)
#             if i < 0:
#                 raise ValueError(f"'{key}' not found in old_dir: {old_dir!r}")
#             old_prefix = old_dir[:i] 
#             j = src_dir.find(self.root_dir)
#             if j < 0:
#                 raise ValueError(f"root_dir_name not in src_dir: {src_dir!r}")
#             src_prefix = src_dir[:j] 

#             path = os.path.join(src_prefix, path)
#             print(f"path: {path}")
#             print(f"old prefix: {old_prefix}")
#             print(f"src_prefix: {src_prefix}")

#             # raise Exception
#             with open(path, 'r') as f:
#                 text = f.read()
#             # replace
#             if old_prefix in text:
#                 new_text = text.replace(old_prefix, src_prefix)
#                 # write back
#                 with open(path, 'w') as f:
#                     f.write(new_text)
#                 print(f"Patched prefixes in {path}")
#         else:
#             print("No match found")

#     def create_dataset(self):
#         # for env_path in self.environments:
#         p = 0 
#         while p < len(self.environments):
#             print("p: ", p)
#             env_path = self.environments[p]
#             try: 
#                 pcl, rgb, depth = self.getEnvironmentPC(env_path)
                
#             except Exception as e:
#                 p -= 1 # Try the same sample again
#                 msg = str(e)
#                 token = "couldn't change to directory"
#                 if token in msg: 
#                     self.fix_g_path(env_path, msg)
#                     continue
#                 else: 
#                     raise Exception

#                 # if "couldn't change to directory " in msg: 


#             env_name = os.path.splitext(os.path.basename(env_path))[0][:-4]
#             path_id = int(re.search(r'data_(\d+)', env_name).group(1))
#             sample_info = self.meta[self.meta["path"] == path_id]
#             task_name = sample_info["task"].iloc[0]
#             tool_name = sample_info["selected_tool"].iloc[0]

#             task_idx = self.label_list_all.index(task_name)
#             tool_idx = self.label_list_all.index(tool_name)

#             print(f"path_id:{path_id} ")
#             print(env_name)
#             point_cloud = np.load(os.path.join(self.root_dir, env_name + "_pc.npy"))
#             print(point_cloud.shape)
#             print(pcl.shape)
#             new_labels, error = self.map_labels(point_cloud[:,:3], point_cloud[:, 3], pcl)
            
#             keep = np.logical_or(new_labels == task_idx,
#                                  new_labels == tool_idx)
            
#             pcl_filt   = pcl[keep]
#             rgb_filt   = rgb[keep]
#             depth_filt = depth[keep]

#             rgb_hm, depth_hm = self.point_cloud_to_heightmap(
#             pcl_filt, rgb_filt, depth_filt,
#             padding=0.005,         # 5 mm extra around the cloud
#             target_size=(224,224)  # or whatever input size your network expects
#             )

#             print("RGB heightmap:", rgb_hm.shape, rgb_hm.dtype)
#             print("Depth heightmap:", depth_hm.shape, depth_hm.dtype)

#             self.save_and_preview_heightmaps(rgb_hm= rgb_hm, depth_hm= depth_hm,path_id=path_id,out_dir= self.save_dir)

#             # pcd = o3d.geometry.PointCloud()
#             # pcd.points = o3d.utility.Vector3dVector(pcl_filt)

#             # # 2) Normalize colors to [0,1] floats
#             # colors = rgb_filt.astype(np.float64)
#             # pcd.colors = o3d.utility.Vector3dVector(colors)

#             # # 3) Visualize
#             # o3d.visualization.draw_geometries([pcd],
#             #     window_name="Filtered Points",
#             #     width=800, height=600,
#             #     left=50, top=50,
#             #     point_show_normal=False)

#             raise Exception
#             p += 1
    
#     def point_cloud_to_heightmap(self, pcl, rgb, depth,
#                                     padding=0.005,
#                                     target_size=(224, 224)):
#         x, y, z = pcl.T
#         H, W     = target_size

#         # 1) Compute XY bounds of your cloud (with padding)
#         x_min, x_max = x.min() - padding, x.max() + padding
#         y_min, y_max = y.min() - padding, y.max() + padding

#         # 2) Compute resolution (meters per pixel)
#         res_x = (x_max - x_min) / W
#         res_y = (y_max - y_min) / H
#         res   = max(res_x, res_y)

#         # 3) Initialize empty maps
#         depth_map = np.full((H, W), -np.inf, dtype=np.float32)
#         rgb_map   = np.zeros((H, W, 3), dtype=np.uint8)

#         # 4) Bin each point
#         for (xi, yi, zi), (ri, gi, bi), di in zip(pcl, rgb, depth):
#             # Convert world → pixel coords
#             col = int((xi - x_min) / res)
#             row = H - 1 - int((yi - y_min) / res)  # flip Y so row 0 is top

#             if 0 <= row < H and 0 <= col < W:
#                 # Keep only the highest point in each cell
#                 if zi > depth_map[row, col]:
#                     depth_map[row, col] = zi
#                     # Map color to 0–255
#                     rgb_map[row, col]   = np.clip(
#                         (np.array([ri, gi, bi]) * 255.0), 0, 255
#                     ).astype(np.uint8)

#         # 5) Replace empty cells’ depth (where we never saw a point)
#         depth_map[depth_map == -np.inf] = 0.0

#         return rgb_map, depth_map
    
#     def save_and_preview_heightmaps(self, rgb_hm: np.ndarray,
#                                 depth_hm: np.ndarray,
#                                 path_id: int,
#                                 out_dir: str):
#         """
#         Save heightmaps as .npy for training, .png for visual inspection,
#         and display them once.
#         """
#         os.makedirs(out_dir, exist_ok=True)
#         # 1) Save raw arrays
#         np.save(os.path.join(out_dir, f"data_{path_id}_rgb.npy"), rgb_hm)
#         np.save(os.path.join(out_dir, f"data_{path_id}_depth.npy"), depth_hm)
        
#         # 2) Save RGB preview as PNG
#         Image.fromarray(rgb_hm).save(
#             os.path.join(out_dir, f"data_{path_id}_rgb.png")
#         )
        
#         # 3) Normalize depth to [0,255] for 8-bit PNG
#         d_min, d_max = depth_hm.min(), depth_hm.max()
#         if d_max > d_min:
#             depth_vis = ((depth_hm - d_min) / (d_max - d_min) * 255).astype(np.uint8)
#         else:
#             depth_vis = np.zeros_like(depth_hm, dtype=np.uint8)
#         Image.fromarray(depth_vis).save(
#             os.path.join(out_dir, f"data_{path_id}_depth.png")
#         )
        
#         # 4) Display once in this notebook
#         plt.figure()
#         plt.title(f"RGB Heightmap (data_{path_id})")
#         plt.imshow(rgb_hm)
#         plt.axis('off')
        
#         plt.figure()
#         plt.title(f"Depth Heightmap (data_{path_id})")
#         plt.imshow(depth_hm, cmap='gray')
#         plt.axis('off')
#         plt.show()
    
#     def map_labels(self, original_pc, original_labels, new_pc):
#         tree = cKDTree(original_pc)
#         dists, idx = tree.query(new_pc, k=1)
#         max_err = dists.max()
#         return original_labels[idx], max_err
    

#     def getEnvironmentPC(self, env_path):
#         C = ry.Config()
#         C.addFile(env_path)

#         table_threshold = C.getFrame("table").getPosition()[2] + C.getFrame("table").info()["size"][2]/2  + 0.005
#         robot_threshold = C.getFrame("l_gripper").getPosition()[2] - 0.04
        

#         pcls, rgbs, depths = [], [], []
#         for cam in self.camera_names:
#             cam_frame = C.getFrame(cam)
#             pcl, rgb, depth = self.getPC_from_cam(cam, C)   
#             pcl = self.cam_to_world(pcl.reshape(-1, 3), cam_frame)

#             pcls.append(pcl)
#             rgbs.append(rgb.reshape(-1, 3))
#             depths.append(depth.reshape(-1))
        
#         all_pcl = np.concatenate(pcls)
#         all_rgb = np.concatenate(rgbs)
#         all_depth = np.concatenate(depths)
        

#         pcl, rgb, depth = self.cleanPointClouds(
#             all_pcl, all_rgb, all_depth,
#             table_threshold, robot_threshold
#         )
#         return pcl, rgb, depth 
        
#     def getPC_from_cam(self, name, C): 
#         cam = ry.CameraView(C)
#         cam.setCamera(name)
#         rgb, depth = cam.computeImageAndDepth(C)
#         pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())

#         return pcl, rgb, depth
    
#     def cam_to_world(self, point_cloud, cam_frame):
#         # Get camera position and rotation matrix
#         t = cam_frame.getPosition()  # Camera position in world frame (1x3 array)
#         R = cam_frame.getRotationMatrix()  # Camera rotation matrix (3x3)

#         # Ensure the point cloud has homogeneous coordinates
#         points_camera_frame_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

#         # Construct the transformation matrix
#         transformation_matrix = np.eye(4)
#         transformation_matrix[:3, :3] = R  # Set the rotation part
#         transformation_matrix[:3, 3] = t   # Set the translation part

#         # Transform points to world frame
#         points_world_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T
#         points_world_frame = points_world_frame_homogeneous[:, :3]

#         return points_world_frame
    
#     def cleanPointClouds(self, pcl, rgb, depth,
#                      table_th, robot_th):
#         # build a single mask for all your z- and y- filters
#         mask = (
#         (pcl[:,2] > table_th) &
#         (pcl[:,2] < robot_th) &
#         (pcl[:,1] > 0) &
#         (pcl[:,1] < 0.6)
#         )
#         pcl   = pcl[mask]
#         rgb   = rgb[mask]
#         depth = depth[mask]
#         return pcl, rgb/256.0, depth



# data_create = ToolBotRAIDataset("minigolf-dataset-top1800-all", "toolbot-minigolf")
# data_create.create_dataset()