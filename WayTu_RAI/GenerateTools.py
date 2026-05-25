import random
import WayTu_RAI.model_utils as mutils
import robotic as ry
import numpy as np
import trimesh

class GenerateTools:     
    
    def __init__(self, cfg, C) -> None:
        self.C = C
        self.cfg = cfg
    
    def getTools(self):
        objs = [self.create_spoon_mesh,
                self.create_sledge_hammer_mesh] # 
        # objs.append(self.create_spatula())
        # objs.append(self.create_hammer())

        return objs

    def set_grasping_waypoint(self, tool_name = None, waypoint = None):
        if waypoint is None:
            tool = self.C.getFrame(tool_name)
            orientation = tool.getQuaternion()
            info = tool.info()
            random_offset = [0.0, -0.15, 0.0]
            position = tool.getPosition() + random_offset 
        else:
            raise Exception("The testing part is currently not implemented.") 
        
        mutils.add_waypoint(self.C, "tool-waypoint", position, orientation)
        print(type(position), position.shape)
        return np.concatenate((position, orientation))

    def set_grasping_waypoint_old(self, tool_name = None, waypoint = None):
        tool = self.C.getFrame(tool_name)
        if waypoint is None:
            orientation = tool.getQuaternion() # mutils.quaternion_rotation(tool.getQuaternion(),90,(0,0,1))
            info = tool.info()

            if info["shape"] == 'ssBox':
                size = np.array(info['size'][:3])
                # Randomize the position of the grasping on tool.
                # random_offset = np.random.uniform(low = -size/2, high = size/2, size= (1,3))
                random_offset = [0.0, 0.0, 0.0]

            position = tool.getPosition() + random_offset          
        else:
            raise Exception("The testing part is currently not implemented.") 
        
        mutils.add_waypoint(self.C, "tool-waypoint", position, orientation)
        print(type(position), position.shape)
        return np.concatenate((position, orientation))
    
    def randomize_shape(self, shape_size, change_factor = 0.1):
        print("shape_size: ", shape_size)
        random_shape = []
        for i in range(len(shape_size)):
            adjustment = shape_size[i] * change_factor
            adjusted_value = shape_size[i] + random.uniform(-1, 1) * adjustment
            random_shape.append(adjusted_value)

        return random_shape
    
    def randomize_number(self, number, change_factor = 0.1):
        print(number)
        adjustment = number * change_factor
        adjusted_value = number + random.uniform(-1, 1) * adjustment

        return adjusted_value

    def create_spoon_mesh(self): 
        # handle
        handle = trimesh.creation.capsule(radius=self.randomize_number(0.3), height=self.randomize_number(1.5), count=[32, 32])

        vertices = handle.vertices.copy()
        upper_half_indices = np.where(vertices[:, 2] > 0)[0]  

        slim_factor = 0.5  
        vertices[upper_half_indices, 0] *= slim_factor  
        vertices[upper_half_indices, 1] *= slim_factor

        handle.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90) , [0, 1, 0]))
        handle.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90) , [0, 0, 1]))

        # Randomize should be added
        donut = trimesh.creation.torus(major_radius= 1.0, minor_radius=0.2,)
        cone = trimesh.creation.cone(radius = 1.01, height = 0.1)

        cone.apply_translation([0, 0, -0.1])

        spoon_bowl = donut.union(cone)

        scaling_matrix = np.diag([0.5, 1, 0.7, 1])
        spoon_bowl.apply_transform(scaling_matrix)

        handle.apply_translation([0, -2.1, 0])
        spoon = spoon_bowl.union(handle)

        scaling_matrix = np.diag([0.08, 0.08, 0.08, 1])
        spoon.apply_transform(scaling_matrix)

        mutils.add_mesh_object(self.C, "spoon", spoon, parent="table", joint=True)

        return "spoon", spoon

    
    def create_sledge_hammer_mesh(self):
        # main block
        rect_block = trimesh.creation.box(extents=self.randomize_shape([0.3, 0.3, 0.3]))

        # sides
        radius, height = self.randomize_number(0.15), self.randomize_number(0.3)
        cylinder_1 = trimesh.creation.cylinder(radius=radius, height=height)
        cylinder_2 = trimesh.creation.cylinder(radius=radius, height=height)

        # handle
        cylinder_3 = trimesh.creation.cylinder(radius=self.randomize_number(0.075), height=self.randomize_number(1))

        h_rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        v_rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])


        cylinder_1.apply_transform(h_rotation_matrix)
        cylinder_2.apply_transform(h_rotation_matrix)
        cylinder_3.apply_transform(h_rotation_matrix)
        cylinder_3.apply_transform(v_rotation_matrix)

        # Position the cylinders at the sides of the rectangular block
        cylinder_1.apply_translation([0.15, 0, 0])
        cylinder_2.apply_translation([-0.15, 0, 0])
        cylinder_3.apply_translation([0, -0.28, 0])

        # Combine the parts into a single hammerhead mesh
        hammer_head = trimesh.util.concatenate([rect_block, cylinder_1, cylinder_2, cylinder_3])
        
        scaling_matrix = np.diag([0.2, 0.2, 0.2, 1])
        hammer_head.apply_transform(scaling_matrix)

        mutils.add_mesh_object(self.C, "sledge-hammer", hammer_head, parent="table", joint = True)

        return "sledge-hammer", hammer_head


    # A function for creating primitive spatula for training the model. 
    def create_spatula(self):
        handle_shape = [0.02, 0.2, 0.02, 0.005]
        head_shape = [0.08, 0.08, 0.01, 0.005]

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, 0.09, 0.0]

        head_position, head_qua = self.add_head_randomness(head_position)

        mutils.add_shape(C=self.C, frame_name="spatula-handle", parent = "table", shape=random_handle_shape, joint = True, color= [1.0, 0.0,0.0])
        mutils.add_shape(C=self.C, frame_name="spatula-head", parent="spatula-handle", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [1.0, 0.0,0.0])

        parameters = {"spatula-handle-shape": [random_handle_shape], 
                      "spatula-head-shape" : [random_head_shape]}

        return "spatula"
    
    # A function for creating primitive spatula for training the model. 
    def create_hammer(self):
        handle_shape = [0.04, 0.25, 0.03, 0.005]
        head_shape = [0.08, 0.04, 0.03, 0.005]
        
        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_head_shape = self.add_shape_randomness(head_shape)

        head_position = [0.0, 0.12, 0.0]
        head_position, head_qua = self.add_head_randomness(head_position)

        mutils.add_shape(C=self.C, frame_name="hammer-handle", parent = "table", shape=random_handle_shape, joint = True, color= [0.0, 1.0,0.0])
        mutils.add_shape(C=self.C, frame_name="hammer-head", parent="hammer-handle", shape = random_head_shape, relative_position= head_position, relative_quaternion=head_qua, color= [0.0, 1.0,0.0])

        parameters = {"hammer-handle-shape": [random_handle_shape], 
                      "hammer-head-shape" : [random_head_shape]}
        
        return "hammer"
    
    # Directly from version 1 - should be checked (probably some hyperparameter tunning is needed)
    def add_shape_randomness(self, shape, change_factor = 0.1):
        random_shape = []
        for i in range(3):
            adjustment = shape[i] * change_factor
            adjusted_value = shape[i] + random.uniform(-1, 1) * adjustment
            random_shape.append(adjusted_value)
        
        random_shape.append(random.uniform(0.001,0.01))

        return random_shape
    
    # Directly from  version 1- should be checked (probably some hyperparameter tunning is needed)
    def add_head_randomness(self, position, change = 0.008, qua = 20):
        random_translated = []
        
        random_translated.append(position[0] + random.uniform(-change,change))
        random_translated.append(position[1] + random.uniform(-change*2,change*2))
        random_translated.append(position[2] + random.uniform(-change,change))
        
        random_angle = random.randint(-qua,qua)
        randomized_qua = mutils.quaternion_rotation([0,0,0,1],random_angle,(0,0,1))

        return random_translated, randomized_qua

    def create_bounding_box(self, C, obj_name, mesh):
        return mutils.create_mesh_bounding_box(self.C, obj_name, mesh)
    


