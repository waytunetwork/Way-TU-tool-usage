import WayTu_RAI.model_utils as mutils

import numpy as np
import robotic as ry 
from scipy.spatial.transform import Rotation as R

class PouringEnvironment: 
    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C
        self.bounding_box_shape = [0.12,0.12,0.1]

    def getObjects(self):
        objs = [self.getPlatform]
        return objs
    
    def getPlatform(self):
        container_width = 0.16
        container_depth = 0.16
        container_height = 0.10

        bottom_thickness = 0.012
        wall_thickness = 0.01

        bottom_shape = [
            container_width,
            container_depth,
            bottom_thickness,
            0.004,
        ]

        side_wall_shape = [
            wall_thickness,
            container_depth,
            container_height,
            0.004,
        ]

        front_wall_shape = [
            container_width,
            wall_thickness,
            container_height,
            0.004,
        ]

        wall_z = bottom_thickness / 2.0 + container_height / 2.0
        wall_x = container_width / 2.0 - wall_thickness / 2.0
        wall_y = container_depth / 2.0 - wall_thickness / 2.0

        container_color = [0.30, 0.65, 0.94]

        mutils.add_shape(
            C=self.C,
            frame_name="pouring-platform",
            parent="table",
            shape=bottom_shape,
            color=container_color,
            joint=True,
            mass=50.0,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pouring-wall-left",
            parent="pouring-platform",
            shape=side_wall_shape,
            color=container_color,
            relative_position=[-wall_x, 0.0, wall_z],
            mass=5.0,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pouring-wall-right",
            parent="pouring-platform",
            shape=side_wall_shape,
            color=container_color,
            relative_position=[wall_x, 0.0, wall_z],
            mass=5.0,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pouring-wall-back",
            parent="pouring-platform",
            shape=front_wall_shape,
            color=container_color,
            relative_position=[0.0, -wall_y, wall_z],
            mass=5.0,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pouring-wall-front",
            parent="pouring-platform",
            shape=front_wall_shape,
            color=container_color,
            relative_position=[0.0, wall_y, wall_z],
            mass=5.0,
        )

        self.C.addFrame("pouring-obj", "pouring-platform")

        return "pouring-platform"

    def update_platform_rotation(self, qua):
            return qua
    
    def getCenter(self): 
            platform_center = self.C.getFrame("pouring-platform").getPosition()
            leg_height = self.C.getFrame("pouring-wall-left").info()["size"][2]
            platform_center[2] += leg_height/2
            return np.array(platform_center)

    def setTargetPosition(self):
        obj = self.C.getFrame("pouring-obj")
        # obj.setPosition(self.C.getFrame("back-joint").getPosition() + [0, 0.2, 0.04])
        obj.setRelativePosition([0, 0.0, 0.1])
        self.obj_pos = obj.getPosition()

    def setTargetQuaternion(self):
        obj = self.C.getFrame("pouring-obj")
        obj.setQuaternion(self.C.getFrame("pouring-platform").getQuaternion())

    def get_waypoints_static(self, C):
        waypoints = {
             "pos" : [],
             "qua" : []
        }
        tool_handle_position = C.getFrame("pitcher-handle-grip").getPosition()
        tool_handle_quaternion = C.getFrame("pitcher-handle-grip").getQuaternion()
        initial_position = self.calculate_initial_pouring_position(
             self.C.getFrame("pouring-platform").getPosition(),
             self.C.getFrame("pitcher-handle-grip").getPosition(),
             0.20,
             self.C.getFrame("pouring-platform").getPosition()[2] + 0.30
        )
        current_mouth_direction = (
        self.C.getFrame("pitcher-spout").getPosition()
            - self.C.getFrame("pitcher-base").getPosition()
        )

        initial_quaternion = self.orient_mouth_toward_target(
            current_quaternion= C.getFrame("pitcher-base").getQuaternion(),
            current_mouth_direction=current_mouth_direction,
            initial_position=initial_position,
            platform_center=C.getFrame("pouring-platform").getPosition(),
        )
        goal_position = self.calculate_initial_pouring_position(
             self.C.getFrame("pouring-platform").getPosition(),
             self.C.getFrame("pitcher-handle-grip").getPosition(),
             0.30,
             self.C.getFrame("pouring-platform").getPosition()[2] + 0.30
        ) + [0.0, 0.0, 0.10]
        goal_quaternion = mutils.rotate_quaternion(
            quaternion = initial_quaternion,
            degrees= -45,
            axis = [1.0, 0.0, 0.0],
        )
        waypoints["pos"].append(tool_handle_position)
        waypoints["qua"].append(tool_handle_quaternion)
        waypoints["pos"].append(initial_position)
        waypoints["qua"].append(initial_quaternion)
        waypoints["pos"].append(goal_position)
        waypoints["qua"].append(goal_quaternion)
        return waypoints

    def calculate_initial_pouring_position(self, 
        platform_center,
        pitcher_center,
        distance_from_platform_center,
        waypoint_height,
    ):
        platform_center = np.asarray(platform_center, dtype=float)
        pitcher_center = np.asarray(pitcher_center, dtype=float)

        # Horizontal direction from the platform toward the pitcher.
        direction = pitcher_center - platform_center
        direction[2] = 0.0

        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-8:
            raise ValueError(
                "Platform and pitcher centers have the same horizontal position."
            )

        direction /= direction_norm

        initial_position = (
            platform_center
            + distance_from_platform_center * direction
        )
        initial_position[2] = waypoint_height

        return initial_position

    
    def orient_mouth_toward_target(
        self,
        current_quaternion,
        current_mouth_direction,
        initial_position,
        platform_center,
    ):
        current_mouth_direction = np.asarray(
            current_mouth_direction,
            dtype=float,
        ).copy()

        # Current horizontal mouth direction.
        current_mouth_direction[2] = 0.0
        current_mouth_direction /= np.linalg.norm(
            current_mouth_direction
        )

        # Desired horizontal mouth direction.
        desired_direction = (
            np.asarray(platform_center, dtype=float)
            - np.asarray(initial_position, dtype=float)
        )
        desired_direction[2] = 0.0
        desired_direction /= np.linalg.norm(desired_direction)

        # Signed yaw correction.
        rotation_angle = np.arctan2(
            np.cross(
                current_mouth_direction,
                desired_direction,
            )[2],
            np.dot(
                current_mouth_direction,
                desired_direction,
            ),
        )

        yaw_correction = R.from_rotvec([
            0.0,
            0.0,
            rotation_angle,
        ])

        current_rotation = R.from_quat(
            current_quaternion,
            scalar_first=True,
        )

        corrected_rotation = (
            yaw_correction * current_rotation
        )

        return corrected_rotation.as_quat(
            scalar_first=True
        )