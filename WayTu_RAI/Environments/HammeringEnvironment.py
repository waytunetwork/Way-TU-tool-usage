import math
import random 
import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation as R

import WayTu_RAI.model_utils as mutils

class HammeringEnvironment: 
    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C

        self.bounding_box_shape = [0.10,0.15,0.20]
    
    def getObjects(self):
        objs = [self.getPlatform]
        return objs
    
    def getPlatform(self):
        hammering_platform_shape = [0.20, 0.25, 0.01, 0.001]
        main_area_shape = [0.20, 0.15, 0.25, 0.001]
        side_area_shape = [0.09, 0.15, 0.02, 0.001]
        top_area_shape = [0.20, 0.15, 0.1, 0.001]

        nail_base_shape = [0.018, 0.15, 0.018, 0.001]
        nail_head_shape = [0.04, 0.01, 0.04, 0.001]

        # The wall creation 
        mutils.add_shape(C=self.C, 
                         frame_name="hammering-platform", 
                         parent = "table", 
                         shape=hammering_platform_shape, 
                         color = (0.76, 0.60, 0.42), 
                         joint=True, 
                         mass= 1000.0)
        
        mutils.add_shape(C=self.C, 
                         frame_name="main-area", 
                         parent = "hammering-platform", 
                         shape=main_area_shape, 
                         color = (0.76, 0.60, 0.42), 
                         relative_position=[0.0, -0.05, 0.13], 
                         mass = 40.0)
        
        mutils.add_shape(C=self.C, 
                         frame_name="left-area", 
                         parent = "main-area", 
                         shape=side_area_shape, 
                         color = (0.76, 0.60, 0.42), 
                         relative_position=[-0.055, 0.0, 0.135], 
                         mass = 20.0)
        mutils.add_shape(C=self.C, 
                         frame_name="right-area", 
                         parent = "main-area", 
                         shape=side_area_shape, 
                         color = (0.76, 0.60, 0.42), 
                         relative_position=[0.055, 0.0, 0.135], 
                         mass = 20.0)
        mutils.add_shape(C=self.C, 
                         frame_name="top-area", 
                         parent = "main-area", 
                         shape=top_area_shape, 
                         color = (0.76, 0.60, 0.42), 
                         relative_position=[0.00, 0.0, 0.195], 
                         mass = 20.0)
        
        # The nail creation
        mutils.add_shape(C=self.C, 
                         frame_name="nail-base", 
                         joint= True ,
                         parent = "hammering-platform",
                         shape=nail_base_shape, 
                         color = [0.3, 0.5, 0.8], 
                         mass= 0.001)
        mutils.add_shape(C=self.C, 
                         frame_name="hammering-obj",
                         parent = "nail-base",
                         shape=nail_head_shape, 
                         color = [0.3, 0.5, 0.8], 
                         relative_position=[0.0, 0.08, 0.0], 
                         mass= 0.001)
        
        return "hammering-platform"
    
    def getCenter(self):
        platform_center = self.C.getFrame("hammering-platform").getPosition()
        wall_height = self.C.getFrame("main-area").info()["size"][2] + self.C.getFrame("left-area").info()["size"][2] + self.C.getFrame("top-area").info()["size"][2]
        platform_center[2] += wall_height/2

        return np.array(platform_center)
    
    def setTargetPosition(self):
        obj = self.C.getFrame("nail-base")
        platform = self.C.getFrame("hammering-platform").getPosition()
        obj.setPosition(platform + [0.0, 0.0, 0.265])
        self.obj_pos = obj.getPosition()
    
    def setTargetQuaternion(self):
        return
    
    def update_platform_rotation(self, qua):
        platform_position = self.C.getFrame("hammering-platform").getPosition()
        print(f"platform position: {platform_position}")
        ang_radian = np.pi / 2  if platform_position[0] > 0 else -np.pi / 2

        print(f"Angle radian: {ang_radian}")
        z_rot = R.from_euler('xyz', [0, 0, ang_radian]).as_quat()
        new_qua = R.from_quat(qua, scalar_first=True) * R.from_quat(z_rot)
        new_qua = new_qua.as_quat(scalar_first=True)

        self.C.getFrame("hammering-platform").setQuaternion(new_qua)
        return new_qua
    
    def get_movement_direction(self):
        base_center = self.C.getFrame("nail-base").getPosition()
        nail_head = self.C.getFrame("hammering-obj").getPosition()
        return (base_center - nail_head) / np.linalg.norm(base_center - nail_head)
    
    def calculate_goal_waypoint(self):
        head_center = self.C.getFrame("hammering-obj").getPosition()
        base_center = self.C.getFrame("nail-base").getPosition()

        direction = head_center - base_center
        direction /= np.linalg.norm(direction)

        distance = np.random.uniform(0.01, 0.05) 

        goal_position = self.C.getFrame("initial-waypoint").getPosition() - direction * distance
        goal_quaternion = self.C.getFrame("initial-waypoint").getQuaternion()

        print("head_center:", head_center)
        print("base_center:", base_center)
        print("direction (before normalization):", head_center - base_center)
        print("direction (normalized):", direction)
        print("initial_pos: ", self.C.getFrame("initial-waypoint").getPosition())
        print("goal_pos:", goal_position)

        return goal_position, goal_quaternion
    
    def target_setup(self):
        target = self.C.getFrame("hammering-obj")
        center_position = target.getPosition()
        base = self.C.getFrame("nail-base")
        nail_center = base.getPosition()
        main_area = self.C.getFrame("main-area")
        main_center = main_area.getPosition()

        return { "hammering-obj" : [center_position],
                 "nail-base": [nail_center],
                 "main-area": [main_center]}
    
    def compute_task_score(self, target_history, max_score=0.50, distance_multiplier=20.0, movement_threshold=0.005):
        # Get center between left and right
        left_center = self.C.getFrame("left-area").getPosition()
        right_center = self.C.getFrame("right-area").getPosition()
        middle_center = (left_center + right_center) / 2

        # Get nail base positions
        base_initial = target_history["nail-base"][0]
        base_final = target_history["nail-base"][-1]

        # Compute movement distance
        movement_distance = np.linalg.norm(base_final - base_initial)

        # If nail didn't move enough, return 0 score
        if movement_distance < movement_threshold:
            return 0.0
        
        platform_positions = target_history["main-area"]
        for i in range(1, len(platform_positions)):
            movement = np.linalg.norm(platform_positions[i] - platform_positions[0])
            if movement > movement_threshold:
                return 0.0  

        # Compute distance to center
        distance_to_center = np.linalg.norm(base_final - middle_center)

        # Exponential decay for score
        score = max_score * np.exp(-distance_multiplier * distance_to_center)
        return score
