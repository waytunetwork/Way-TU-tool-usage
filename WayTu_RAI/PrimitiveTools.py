import random
import robotic as ry
import numpy as np

import WayTu_RAI.model_utils as mutils
from scipy.spatial.transform import Rotation as R


# 27.01: The primitive tools that will be used in training
# 27.01: Hammer, Spatula, L-ruler, Stick, Knife, Wrench

class GeneratePrimitiveTools: 
    def __init__(self, cfg, C):
        self.C = C
        self.cfg = cfg
    
    def getTools(self):
        objs = [ 
            self.create_L_ruler,
            self.create_hammer,
            self.create_spatula,
            # self.create_screwdriver,
            # self.create_ball,
            # self.create_rolling_pin,

            ]
        
        # self.create_screwdriver
        # self.create_wrench
        # self.create_hammer,
        # self.create_spatula, 
        # self.create_L_ruler, 
        return objs

    # Random offset is currently is not random
    def set_grasping_waypoint(self, tool_name = None, waypoint = None):
        if waypoint is None:
            tool_name = tool_name + '-handle'
            tool = self.C.getFrame(tool_name)
            orientation = tool.getQuaternion()
            info = tool.info()
            random_offset = [0.0, 0.0, 0.0] # [0.0, -0.10, 0.0]
            position = tool.getPosition() + random_offset 
        else:
            raise Exception("The testing part is currently not implemented.") 
        
        mutils.add_waypoint(self.C, "tool-waypoint", position, orientation)
        return np.concatenate((position, orientation))
    
    def create_g_L_ruler(self):
        self.C.addFile("/home/ece/git/WayTU-002/L_ruler_draft.g")
        return "L-ruler"
    
    def create_wrench(self):
        base_shape = [0.02, 0.20, 0.02, 0.004]
        head_01_shape = [0.08, 0.02, 0.02, 0.004]
        head_02_shape = [0.02, 0.06, 0.02, 0.004]

        random_base_shape = self.add_shape_randomness(base_shape)
        random_head_01_shape = self.add_shape_randomness(head_01_shape)
        random_head_02_shape = self.add_shape_randomness(head_02_shape)

        # head_position = [0.0, 0.10, 0.0]
        # head_position, head_qua = self.add_head_randomness(head_position)
        
        head_position = [0.0, random_base_shape[1]/2 + random_head_01_shape[1]/2 - 0.01, 0.0]

        mutils.add_shape(C=self.C, frame_name="wrench-base", parent = "table", shape=random_base_shape, joint = True, color= [0.5, 0.5, 0.5])
        mutils.add_shape(C=self.C, frame_name="wrench-head-01", parent="wrench-base", shape = random_head_01_shape, relative_position= head_position, color= [0.5, 0.5, 0.5])
        mutils.add_shape(C=self.C, frame_name="wrench-head-02", parent="wrench-head-01", shape = random_head_02_shape, position= self.C.getFrame("wrench-head-01").getPosition() - [random_head_01_shape[0]/2, - random_head_02_shape[1]/2, 0.0], color= [0.5, 0.5, 0.5])
        mutils.add_shape(C=self.C, frame_name="wrench-head-03", parent="wrench-head-01", shape = random_head_02_shape, position= self.C.getFrame("wrench-head-01").getPosition() + [random_head_01_shape[0]/2, + random_head_02_shape[1]/2, 0.0], color= [0.5, 0.5, 0.5])

        return "wrench", ["wrench-base", "wrench-head-01", "wrench-head-02", "wrench-head-03"]

    # 07.02: Updated to be in the same format with the realistic objects
    def create_hammer(self):
        handle_shape = [0.035, 0.25, 0.035, 0.008]
        head_shape = [0.07, 0.04, 0.04, 0.01]
        
        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, 0.12, 0.0]
        head_position, head_qua = self.add_head_randomness(head_position)

        # mutils.add_shape(C=self.C, frame_name="hammer-base", parent = "table", shape=random_handle_shape, joint = True, color= [0.0, 1.0,0.0])
        # mutils.add_shape(C=self.C, frame_name="hammer-head", parent="hammer-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.0, 1.0,0.0])
        mutils.add_shape(C=self.C, frame_name="hammer-base", parent = "table", shape=random_handle_shape, joint = True, color= [0.55, 0.27, 0.07])
        mutils.add_shape(C=self.C, frame_name="hammer-head", parent="hammer-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.7, 0.7, 0.75])

                                                                                                                                                        
        return "hammer" # , ["hammer-handle", "hammer-head"]
    
    def create_rolling_pin(self):
        handle_shape = [0.002, 0.002, 0.18, 0.02]
        head_shape = [0.001, 0.001, 0.25, 0.01]

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, 0.0, 0.0]
        head_position, head_qua = self.add_head_randomness(head_position)

        mutils.add_shape(C=self.C, frame_name="rolling-pin-base", parent = "table", shape=random_handle_shape, joint = True, color= [0.0, 1.0,0.0])
        mutils.add_shape(C=self.C, frame_name="rolling-pin-head", parent="rolling-pin-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.0, 1.0,0.0])


        return "rolling-pin"
    
    def create_thin_stick(self):
        return "thin-stick"
    
    def create_plus_shape(self):
        return "plus"

    def create_knife(self):
        handle_shape    = [0.03, 0.10, 0.03, 0.007]
        head1_shape     = [0.02, 0.15, 0.01, 0.002]
        head2_shape     = [0.02, 0.10, 0.01, 0.002]

        head1_position = [-0.01, 0.09, 0.0]
        head1_qua = [1.0, 0.0, 0.0, 0.0]

        

        

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head1_shape)

        
        # head1_position, head1_qua = self.add_head_randomness(head1_position, qua = 5)

        mutils.add_shape(C=self.C, frame_name="knife-base", parent = "table", shape=random_handle_shape, joint = True, color= [1.0, 0.0,0.0])
        mutils.add_shape(C=self.C, frame_name="knife-head-1", parent="knife-base", shape = random_head_shape, relative_position= head1_position, relative_quaternion=head1_qua, color= [1.0, 0.0,0.0])
        
        
        random_angle = 25 # np.random.uniform(-20, 20)
        angle_radians = np.radians(random_angle)
        
        
        pivot_position = np.array([0, random_head_shape[1]/2, 0])


        direction_vector = np.array([
            head2_shape[1]/2 * np.sin(angle_radians),  
            head1_shape[1]/2 - head2_shape[1]/2 * np.cos(angle_radians),  
            0  
        ])

        print(direction_vector)

        head2_position = direction_vector

        rotation_axis = np.array([0, 0, 1])  # Z-axis rotation
        rotation = R.from_rotvec(angle_radians * rotation_axis)
        head2_quaternion = rotation.as_quat(scalar_first=True)
        
        mutils.add_shape(C=self.C, frame_name="knife-head-2", parent="knife-head-1", shape = head2_shape, relative_position= head2_position, relative_quaternion=head2_quaternion, color= [1.0, 0.0,0.0])

        return "knife"

    def create_screwdriver(self): 
        handle_shape    = [0.03, 0.15, 0.03, 0.007]
        head_shape      = [0.010, 0.15, 0.015, 0.007]   

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, -0.075, 0.0]
        head_position, head_qua = self.add_head_randomness(head_position, qua = 5)

        mutils.add_shape(C=self.C, frame_name="screwdriver-base", parent = "table", shape=random_handle_shape, joint = True, color= [0.0, 1.0,0.0])
        mutils.add_shape(C=self.C, frame_name="screwdriver-head", parent="screwdriver-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.0, 1.0,0.0])

                                                                                                                                                        
        return "screwdriver"
    
    # Distractor object number 1
    def create_ball(self):
        handle_shape = [0.06, 0.06, 0.06, 0.05]
        random_handle_shape = self.add_shape_randomness(handle_shape)
        # print(f"###DEBUG### random handle shape {random_handle_shape}")
        mutils.add_shape(C=self.C, frame_name="ball-base", parent = "table", shape=random_handle_shape, joint = True, color= [1.0, 0.0,0.0])

        return "ball"
    
    # 10.02: Updated like hammer
    def create_spatula(self):
        handle_shape = [0.025, 0.25, 0.025, 0.005]
        head_shape = [0.08, 0.08, 0.01, 0.005]

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, 0.09, 0.0]

        head_position, head_qua = self.add_head_randomness(head_position)

        # mutils.add_shape(C=self.C, frame_name="spatula-base", parent = "table", shape=random_handle_shape, joint = True, color= [1.0, 0.0,0.0])
        # mutils.add_shape(C=self.C, frame_name="spatula-head", parent="spatula-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [1.0, 0.0,0.0])
        mutils.add_shape(C=self.C, frame_name="spatula-base", parent = "table", shape=random_handle_shape, joint = True, color= [0.15, 0.15, 0.15])
        mutils.add_shape(C=self.C, frame_name="spatula-head", parent="spatula-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.8, 0.8, 0.8])
        return "spatula" # , ["spatula-handle", "spatula-head"]
    
    def create_L_ruler(self):
        handle_shape = [0.04, 0.25, 0.02, 0.005]
        head_shape =  [0.15, 0.04, 0.02, 0.005]

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        offset_x = random.choice([-0.06, 0.06])
        offset_y = random.choice([-0.1, 0.1])
        head_position = [offset_x, offset_y, 0.0]
        head_position, head_qua = self.add_head_randomness(head_position)
        head_qua = mutils.quaternion_rotation(head_qua, -180, [0,0,1])

        # mutils.add_shape(C=self.C, frame_name="L-ruler-base", parent = "table", shape=random_handle_shape, color= [0.65, 0.5, 0.39])
        # mutils.add_shape(C=self.C, frame_name="L-ruler-head", parent="L-ruler-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.65, 0.5, 0.39])
        mutils.add_shape(C=self.C, frame_name="L-ruler-base", parent = "table", shape=random_handle_shape, color= [0.65, 0.65, 0.7])
        mutils.add_shape(C=self.C, frame_name="L-ruler-head", parent="L-ruler-base", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.65, 0.65, 0.7])


        return "L-ruler" # , ["L-ruler-handle", "L-ruler-head"]

    def add_shape_randomness(self, shape, change_factor = 0.1):
        random_shape = []
        for i in range(3):
            adjustment = shape[i] * change_factor
            adjusted_value = shape[i] + random.uniform(-1, 1) * adjustment
            random_shape.append(adjusted_value)
        
        # random_shape.append(random.uniform(0.001,0.01))
        random_shape.append(shape[3])
        return random_shape
    
    # Directly from  version 1- should be checked (probably some hyperparameter tunning is needed)
    def add_head_randomness(self, position, change = 0.008, qua = 20):
        random_translated = []
        
        random_translated.append(position[0] + random.uniform(-change,change))
        random_translated.append(position[1] + random.uniform(-change*2,change*2))
        random_translated.append(position[2] + random.uniform(-change*0.1,change*0.5))
        
        random_angle = random.randint(-qua,qua)
        randomized_qua = mutils.quaternion_rotation([0,0,0,1],random_angle,(0,0,1))

        return random_translated, randomized_qua
    
    def create_bounding_box(self,C, obj_name, obj_part_names):
        return mutils.create_primitive_bounding_box(C, obj_part_names)
    
 
