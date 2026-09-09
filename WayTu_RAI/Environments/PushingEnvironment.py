import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation as R

import WayTu_RAI.model_utils as mutils


class PushingEnvironment:
    """Simple planar pushing task for zero-shot minigolf-model tests.

    The physical scene contains a flat platform, a movable target object, and a
    non-colliding goal marker.  Any point-cloud masking or substitution used to
    present this task to the pretrained minigolf model should be performed in
    the inference/test code, not in this environment class.
    """

    PLATFORM_NAME = "pushing-platform"
    TARGET_NAME = "pushing-obj"
    GOAL_NAME = "pushing-goal"

    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C

        self.platform_shape = np.asarray(
            cfg.get("pushing-platform-shape", [0.25, 0.35, 0.01, 0.005]),
            dtype=float,
        )
        self.target_shape = np.asarray(
            cfg.get("pushing-target-shape", [0.05, 0.05, 0.05, 0.01]),
            dtype=float,
        )
        self.goal_shape = np.asarray(
            cfg.get("pushing-goal-shape", [0.05, 0.05, 0.004, 0.002]),
            dtype=float,
        )

        self.bounding_box_shape = self.platform_shape[:3].tolist()

    def getObjects(self):
        return [self.getPlatform]

    def getPlatform(self):
        mutils.add_shape(
            C=self.C,
            frame_name=self.PLATFORM_NAME,
            parent="table",
            shape=self.platform_shape.tolist(),
            color=[0.85, 0.85, 0.85],
            joint=True,
            mass=50.0,
        )

        goal_y = (
            self.platform_shape[1] / 2
            - self.goal_shape[1] / 2
            - 0.025
        )
        goal_z = self.platform_shape[2] / 2 + self.goal_shape[2] / 2
        mutils.add_shape(
            C=self.C,
            frame_name=self.GOAL_NAME,
            parent=self.PLATFORM_NAME,
            shape=self.goal_shape.tolist(),
            color=[0.2, 0.8, 0.2, 1.0],
            relative_position=[0.0, -goal_y, goal_z],
            mass=0.0,
        )
        self.C.getFrame(self.GOAL_NAME).setContact(0)

        mutils.add_shape(
            C=self.C,
            frame_name=self.TARGET_NAME,
            parent=self.PLATFORM_NAME,
            shape=self.target_shape.tolist(),
            color=[0.1, 0.3, 1.0],
            joint=True,
            mass=float(self.cfg.get("pushing-target-mass", 0.05)),
        )

        return self.PLATFORM_NAME

    def get_movement_direction(self):
        goal_position = self.C.getFrame(self.GOAL_NAME).getPosition()
        target_position = self.C.getFrame(self.TARGET_NAME).getPosition()
        direction = goal_position - target_position
        direction[2] = 0.0

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.zeros(3)
        return direction / norm

    def get_goal(self):
        goal_position = self.C.getFrame(self.GOAL_NAME).getPosition().copy()
        goal_position[2] = self.C.getFrame("initial-waypoint").getPosition()[2]
        return goal_position

    def calculate_goal_waypoint(self):
        initial_waypoint = self.C.getFrame("initial-waypoint")
        goal_position = self.get_goal()
        goal_quaternion = initial_waypoint.getQuaternion()
        return goal_position, goal_quaternion

    def target_setup(self):
        target_position = self.C.getFrame(self.TARGET_NAME).getPosition().copy()
        return {self.TARGET_NAME: [target_position]}

    def compute_task_score(
        self,
        target_history,
        max_score=0.5,
        decay_factor=1.0,
        movement_threshold=0.01,
    ):
        initial_position = np.asarray(target_history[self.TARGET_NAME][0])
        final_position = np.asarray(target_history[self.TARGET_NAME][-1])

        if np.linalg.norm(final_position - initial_position) < movement_threshold:
            return 0.0

        goal_frame = self.C.getFrame(self.GOAL_NAME)
        goal_center = goal_frame.getPosition()
        goal_size = np.asarray(goal_frame.info()["size"][:3])
        goal_quaternion = goal_frame.getQuaternion()
        goal_rotation = R.from_quat(
            goal_quaternion, scalar_first=True
        ).as_matrix()

        target_local = goal_rotation.T @ (final_position - goal_center)
        half_size_xy = goal_size[:2] / 2
        distance_vector_xy = np.maximum(
            np.abs(target_local[:2]) - half_size_xy,
            0.0,
        )
        distance_to_goal = np.linalg.norm(distance_vector_xy)

        task_score = max_score - decay_factor * distance_to_goal
        return float(np.clip(task_score, 0.0, max_score))

    def update_platform_rotation(self, qua):
        return qua

    def getCenter(self):
        platform_center = self.C.getFrame(self.PLATFORM_NAME).getPosition().copy()
        platform_center[2] += (
            self.platform_shape[2] + self.target_shape[2]
        ) / 2
        return platform_center

    def setTargetPosition(self):
        target = self.C.getFrame(self.TARGET_NAME)
        target.setPosition(self.position_target())
        self.obj_pos = target.getPosition().copy()

    def position_target(self):
        platform = self.C.getFrame(self.PLATFORM_NAME)
        platform_center = platform.getPosition()
        platform_quaternion = platform.getQuaternion()

        margin_x = self.target_shape[0] / 2 + 0.02
        x_local = np.random.uniform(
            -self.platform_shape[0] / 2 + margin_x,
            self.platform_shape[0] / 2 - margin_x,
        )

       # Keep the object in the positive-y half of the platform.
        margin_y = self.target_shape[1] / 2 + 0.02
        y_local = np.random.uniform(
            margin_y,
            self.platform_shape[1] / 2 - margin_y,
        )
        z_local = self.platform_shape[2] / 2 + self.target_shape[2] / 2

        local_position = np.array([x_local, y_local, z_local])
        rotation_matrix = R.from_quat(
            platform_quaternion, scalar_first=True
        ).as_matrix()
        return platform_center + rotation_matrix @ local_position

    def setTargetQuaternion(self):
        return
