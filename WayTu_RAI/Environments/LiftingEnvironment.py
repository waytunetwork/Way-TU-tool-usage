import robotic as ry
import random
import WayTu_RAI.model_utils as mutils
import time
import numpy as np
import math

class LiftingEnvironment:
    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C

        self.bounding_box_shape = [0.15,0.25,0.2]
    
    def getObjects(self):
        objs = [self.getPlatform]
        return objs
    
    def getPlatform(self):
        plate_shape = [0.2,0.4,0.04,0.005]
        leg_shape = [0.04,0.04,0.4,0.005]
        joint_shape = [0.06,0.04,0.04,0.005]
        joint_base, random_height = self.get_platform_height(True)
        obj_shape = [0.03,0.5,0.03,0.005]

        
        mutils.add_shape(C=self.C, frame_name="lifting-platform", parent = "table", shape=plate_shape, color = [.1], joint=True, mass= 50.0)

        mutils.add_shape(C=self.C, frame_name="back-leg1", parent = "lifting-platform", shape=leg_shape, color = [.1], relative_position=[-0.04, -0.2, 0.20], mass = 5.0)
        mutils.add_shape(C=self.C, frame_name="back-joint", parent = "back-leg1", shape=joint_shape, color = [.1], relative_position=joint_base,  mass = 5.0)
        mutils.add_shape(C=self.C, frame_name="back-leg2", parent = "back-joint", shape=leg_shape, color = [.1], relative_position=[0.04, 0, -0.1 - random_height],  mass = 5.0)

        mutils.add_shape(C=self.C, frame_name="front-leg1", parent = "lifting-platform", shape=leg_shape, color = [.1], relative_position=[-0.04, 0.2, 0.20],  mass = 5.0)
        mutils.add_shape(C=self.C, frame_name="front-joint", parent = "front-leg1", shape=joint_shape, color = [.1], relative_position=joint_base,  mass = 5.0)
        mutils.add_shape(C=self.C, frame_name="front-leg2", parent = "front-joint", shape=leg_shape, color = [.1], relative_position=[0.04, 0, -0.1 - random_height],  mass = 5.0)

        mutils.add_shape(C=self.C, frame_name="lifting-obj", joint= True ,parent = "back-joint", shape=obj_shape, color = [0, 0, 1.0], mass= 0.00001) # [0.35,0.98,1.0]


        return "lifting-platform"
    
    def get_movement_direction(self):
        return np.array([0, 0, 1]) 

    def get_goal(self):
        height_offset = np.random.uniform(0.05, 0.15)
        return np.array([0, 0, height_offset])
    
    def calculate_goal_waypoint(self):
        initial_waypoint = self.C.getFrame("initial-waypoint")
        initial_position = initial_waypoint.getPosition()
        initial_quaternion = initial_waypoint.getQuaternion()

        goal_position = initial_position + self.get_goal()
        goal_quaternion = initial_quaternion

        return goal_position, goal_quaternion
    
    def target_setup(self):
        target = self.C.getFrame("lifting-obj")
        center_position = target.getPosition()
        transform = target.getTransform()
        size = target.info()['size'][:3]

        local_offset_1 = np.array([size[0]/2, size[1]/2, size[2]/2, 1]) 
        local_offset_2 = np.array([-size[0]/2, -size[1]/2, -size[2]/2, 1])

        end_point_1 = transform @ local_offset_1
        end_point_2 = transform @ local_offset_2
    
        self.C.addFrame("end-point-1", "lifting-obj")\
                .setShape(ry.ST.marker, [.1])\
                .setPosition(end_point_1[:3])
            
        self.C.addFrame("end-point-2", "lifting-obj")\
                .setShape(ry.ST.marker, [.1])\
                .setPosition(end_point_2[:3])
        
        return { "end-point-1": [end_point_1],
                "end-point-2": [end_point_2],
                "lifting-obj" : [center_position] }
    
    def compute_task_score(self, target_history):
        stability_weight = 5
        lift_scaling = 10

        h_initial = target_history["lifting-obj"][0][2]  
        h_final = target_history["lifting-obj"][-1][2]  
        lift_amount = h_final - h_initial

        lift_score = 1 - np.exp(-lift_scaling * lift_amount)

        height_diff = np.abs((target_history["end-point-1"][-1][2] - target_history["end-point-1"][0][2]) -
                            (target_history["end-point-2"][-1][2] - target_history["end-point-2"][0][2]))
        
        stability_score = 1 / (1 + stability_weight * height_diff)
        task_score = lift_score * stability_score

        return task_score
    
    def getCenter(self): 
        platform_center = self.C.getFrame("lifting-platform").getPosition()
        leg_height = self.C.getFrame("back-leg1").info()["size"][2]
        platform_center[2] += leg_height/2
        return np.array(platform_center)

    def setTargetPosition(self):
        obj = self.C.getFrame("lifting-obj")
        # obj.setPosition(self.C.getFrame("back-joint").getPosition() + [0, 0.2, 0.04])
        obj.setRelativePosition([0, 0.2, 0.04])
        self.obj_pos = obj.getPosition()
        
    def setTargetQuaternion(self):
        obj = self.C.getFrame("lifting-obj")
        obj.setQuaternion(self.C.getFrame("lifting-platform").getQuaternion())

       
    def get_platform_height(self, rand = False):
        joint_relative_position_base = [0.04, 0, 0.1]
        random_height = 0

        if rand == True: 
            random_height = random.uniform(-2, 1) * 0.05
            joint_relative_position_base [2] += + random_height

        return  joint_relative_position_base, random_height
    
    def get_training_waypoints(self):
        return self.get_lifting_waypoints_static()
    
    
    def get_score(self, tool_old, tool_new, obj_new):
        grasp_distance =  math.sqrt(((tool_new[0] - tool_old[0]) ** 2)*0.5 + ((tool_new[1] - tool_old[1]) ** 2)*0.5+ ((tool_new[2] - tool_old[2]) ** 2)) 
        print("grasp_distance:  ", grasp_distance)
        grasp_score = 1 if grasp_distance > 0.45  else 0

        obj_score = math.sqrt((obj_new[0] - self.obj_pos[0]) ** 2 + (obj_new[1] - self.obj_pos[1]) ** 2+ (obj_new[2] - self.obj_pos[2]) ** 2)
        print("obj_score: ", obj_score)
        lift_score =  0.4 * grasp_score + 0.6 * obj_score

        return lift_score

    def getTargetName(self):
        return "lifting-platform"
    
    def update_platform_rotation(self, qua):
        return qua

    
    def get_lifting_waypoints_static(self):
        gripper_pos = self.C.getFrame("l_gripper").getPosition() 
        target_pos = self.C.getFrame("lifting-obj").getPosition()

        side = - 1 if gripper_pos[0] < target_pos[0] else 1
        distance_x = random.uniform(0.05, 0.15) * side 
        
        height = 0.1
        x_bound = 0.0

        degree = 90

        obj_shape = self.C.getFrame("lifting-obj").info()['size'][:3]

        y_bound = np.random.uniform(-obj_shape[1]/2.5,obj_shape[1]/2.5)
        z_bound = np.random.uniform(-obj_shape[2]/2.1,obj_shape[2]/2.1)
        # x_bound = np.random.uniform(-distance_x/2.1,distance_x/2.1)

        initial_pos = self.C.getFrame("lifting-obj").getPosition() + [distance_x + x_bound,y_bound,  -height + z_bound]
        initial_qua = mutils.quaternion_rotation(self.C.getFrame("lifting-obj").getQuaternion(), degree, [0, 0, 1])
        mutils.add_waypoint(self.C, "initial-waypoint", initial_pos, initial_qua)

        y_bound = np.random.uniform(-obj_shape[1]/2.5,obj_shape[1]/2.5)
        z_bound = np.random.uniform(-obj_shape[2]/2.1,obj_shape[2]/2.1)
        # x_bound = np.random.uniform(-distance_x/2.1,distance_x/2.1)
        
        goal_pos = self.C.getFrame("lifting-obj").getPosition() + [distance_x + x_bound, y_bound, height*2 + z_bound]
        goal_qua = mutils.quaternion_rotation(self.C.getFrame("lifting-obj").getQuaternion(), degree, [0, 0, 1])
        mutils.add_waypoint(self.C, "goal-waypoint", goal_pos, goal_qua)

        return np.concatenate((initial_pos,initial_qua)), np.concatenate((goal_pos, goal_qua))
    
    def get_lifting_waypoints(self):
        distance = 0.1
        height = 0.05
        target_position = self.C.getFrame("lifting-obj").getPosition()
        robot_base = self.C.getFrame("l_panda_base").getPosition()

        direction = robot_base - target_position
        norm_dir = direction / np.linalg.norm(direction)
        dist_dir = norm_dir * distance

        mid_target= target_position + dist_dir

        target_qua = self.C.getFrame("lifting-obj").getQuaternion()
        
        initial_pos = mid_target + [0.0, 0.0, -height]
        goal_pos = mid_target + [0.0, 0.0, height*1.5] 

        mutils.add_waypoint(self.C, "initial-waypoint", initial_pos, target_qua)
        mutils.add_waypoint(self.C, "goal-waypoint", goal_pos, target_qua)


    
