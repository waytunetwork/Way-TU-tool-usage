import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation as R

import WayTu_RAI.model_utils as mutils

class MinigolfEnvironment: 
    def __init__(self, cfg, C):
        self.cfg = cfg 
        self.C = C

        self.bounding_box_shape = [0.15,0.30,0.15]

    def getObjects(self):
        objs = [self.getPlatform]
        return objs
    
    def getPlatform(self):
        minigolf_platform_shape =   [0.26, 0.4 ,0.01, 0.005]
        main_area_shape         =   [0.26, 0.2,0.1,0.005]
        left_area_shape         =   [0.06, 0.1, 0.1,0.005]
        right_area_shape        =   [0.06, 0.1,0.1,0.005]
        end_area_shape          =   [0.26, 0.1,0.1,0.005]
        minigolf_obj_shape      =   [0.05, 0.05, 0.05, 0.02]

        mutils.add_shape(C=self.C, frame_name="minigolf-platform", parent = "table", shape=minigolf_platform_shape, color = [1.0, 1.0, 1.0], joint=True, mass= 50.0)
        
        mutils.add_shape(C=self.C, frame_name="main-area", parent = "minigolf-platform", shape=main_area_shape, color = [.1], relative_position=[0.0, 0.1, 0.0505], mass = 40.0)
        mutils.add_shape(C=self.C, frame_name="left-area", parent = "main-area", shape=left_area_shape, color = [.1], relative_position=[0.1, -0.15, 0], mass = 20.0)
        mutils.add_shape(C=self.C, frame_name="right-area", parent = "main-area", shape=right_area_shape, color = [.1], relative_position=[-0.1, -0.15, 0], mass = 20.0)
        mutils.add_shape(C=self.C, frame_name="end-area", parent = "main-area", shape=end_area_shape, color = [.1], relative_position=[0, -0.25, 0], mass = 30.0)

        mutils.add_shape(C=self.C, frame_name="minigolf-obj", joint= True ,parent = "minigolf-platform", shape=minigolf_obj_shape, color = [0, 0, 1.0], mass= 0.00001) # [0.35,0.98,1.0]

        return "minigolf-platform"

    def get_movement_direction(self):
        hole_center = (self.C.getFrame("left-area").getPosition() + self.C.getFrame("right-area").getPosition()) / 2
        ball_pos = self.C.getFrame("minigolf-obj").getPosition()
        direction = hole_center - ball_pos
        direction[2] = 0  # Keep horizontal
        return direction / np.linalg.norm(direction)
    
    def get_goal(self):
        hole_center = (self.C.getFrame("left-area").getPosition() + self.C.getFrame("right-area").getPosition())/2
        hole_center[2] = self.C.getFrame("initial-waypoint").getPosition()[2]

        return hole_center
    
    def calculate_goal_waypoint(self):
        initial_waypoint = self.C.getFrame("initial-waypoint")
        initial_position = initial_waypoint.getPosition()
        initial_quaternion = initial_waypoint.getQuaternion()

        goal_position = self.get_goal()
        goal_quaternion = initial_quaternion

        return goal_position, goal_quaternion
    
    def target_setup(self):
        target = self.C.getFrame("minigolf-obj")
        center_position = target.getPosition()

        return { "minigolf-obj" : [center_position]}

    import numpy as np

    def compute_task_score(self, target_history, max_score=0.5, decay_factor=1, movement_threshold=0.01):
        hole_center, hole_size, hole_transform = self.compute_hole_position()

        # Get initial and final ball positions
        initial_position = target_history["minigolf-obj"][0]  # First recorded position
        final_position = target_history["minigolf-obj"][-1]   # Latest recorded position

        # Compute movement distance
        movement_distance = np.linalg.norm(final_position - initial_position)

        # If the ball did not move significantly, return 0
        if movement_distance < movement_threshold:
            return 0.0  # No movement → 0 score

        # Compute ball position relative to the hole
        ball_local_position = np.linalg.inv(hole_transform) @ (final_position - hole_center)
        
        half_size = hole_size / 2
        distance_vector = np.maximum(np.abs(ball_local_position) - half_size, 0)
        distance_to_hole = np.linalg.norm(distance_vector)

        # Compute task score (decay based on distance)
        task_score = max_score - decay_factor * distance_to_hole

        return task_score

    def update_platform_rotation(self, qua):
        return qua

    def compute_hole_position(self):

    # Get positions of reference areas
        main_area_center = self.C.getFrame("main-area").getPosition()
        left_area_center = self.C.getFrame("left-area").getPosition()
        right_area_center = self.C.getFrame("right-area").getPosition()
        
        main_area_shape = self.C.getFrame("main-area").info()["size"][:3]
        left_area_shape = self.C.getFrame("left-area").info()["size"][:3]
        end_area_shape = self.C.getFrame("end-area").info()["size"][:3]
        right_area_shape = self.C.getFrame("right-area").info()["size"][:3]



        # Get platform rotation
        platform_quaternion = self.C.getFrame("minigolf-platform").getQuaternion()
        
        # Compute the hole's local position (midpoint of left & right areas)
        hole_local_x = (left_area_center[0] + right_area_center[0]) / 2
        hole_local_y = (left_area_center[1] + right_area_center[1]) / 2
        hole_local_z = main_area_center[2] - (0.5 * main_area_shape[2])  # At the bottom of the main area

        hole_local_position = np.array([hole_local_x, hole_local_y, hole_local_z])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(platform_quaternion, scalar_first=True).as_matrix()
        
        # Transform the hole's local position to global coordinates
        hole_global_position = main_area_center + rotation_matrix @ (hole_local_position - main_area_center)

        hole_width = abs(left_area_center[0] - right_area_center[0]) - left_area_shape[0]/2 - right_area_shape[0]/2
        hole_depth = self.C.getFrame("main-area").info()["size"][2] - main_area_shape[1]/2 - end_area_shape[1]/2
        hole_height = main_area_shape[2]

        return hole_global_position, np.array([hole_width,hole_depth, hole_height]), rotation_matrix


    def getCenter(self):
        # the minigolf platform is on the table, we need to calculate its center
        platform_center = self.C.getFrame("minigolf-platform").getPosition()
        wall_height = self.C.getFrame("main-area").info()["size"][2] + self.C.getFrame("minigolf-obj").info()["size"][2]
        platform_center[2] += wall_height/2
        return np.array(platform_center)
    
    def setTargetPosition(self):
        obj = self.C.getFrame("minigolf-obj")
        obj.setPosition(self.position_target())
        self.obj_pos = obj.getPosition()
    
    def position_target(self):
        main_area_center = self.C.getFrame("main-area").getPosition()
        main_area_shape = self.C.getFrame("main-area").info()["size"]
        main_area_quaternion = self.C.getFrame("minigolf-platform").getQuaternion()

        x_local = np.random.uniform(-main_area_shape[0] / 2 + 0.02, main_area_shape[0] / 2 - 0.02)
        y_local = np.random.uniform(-main_area_shape[1] / 2 + 0.02, main_area_shape[1] / 2 - 0.02)
        z_local = main_area_shape[2]/2 + self.C.getFrame("minigolf-obj").info()["size"][2]/2 

        local_position = np.array([x_local, y_local, z_local])
        rotation_matrix = R.from_quat(main_area_quaternion, scalar_first=True).as_matrix()

        global_position = main_area_center + rotation_matrix @ local_position

        return global_position
    
    def setTargetQuaternion(self):
        return