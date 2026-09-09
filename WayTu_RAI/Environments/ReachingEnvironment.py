import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation as R

import WayTu_RAI.model_utils as mutils


class ReachingEnvironment:
    """Obstacle-constrained reaching environment with a fixed wall opening."""

    def __init__(self, cfg, C):
        self.cfg = cfg
        self.C = C

        self.platform_size = np.array([0.30, 0.30, 0.01])
        self.wall_width = 0.28
        self.wall_depth = 0.1
        self.wall_height = 0.18
        self.opening_width = 0.11
        self.opening_height = 0.10
        self.target_size = np.array([0.05, 0.04, 0.04])
        self.riser_height = 0.07

        # Used by GenerateEnvironment when checking scene placement.
        self.bounding_box_shape = [0.15, 0.15, 0.21]

    def getObjects(self):
        return [self.getPlatform]

    def getPlatform(self):
        # Keep the main platform thin.
        platform_shape = [*self.platform_size, 0.002]

        # Riser between the main platform and the task elements.
        riser_shape = [
            self.platform_size[0],
            self.platform_size[1],
            self.riser_height,
            0.002,
        ]

        side_width = (self.wall_width - self.opening_width) / 2.0
        side_shape = [
            side_width,
            self.wall_depth,
            self.wall_height,
            0.002,
        ]

        top_height = self.wall_height - self.opening_height
        top_shape = [
            self.wall_width,
            self.wall_depth,
            top_height,
            0.002,
        ]

        target_shape = [*self.target_size, 0.002]

        # Height of the surface supporting all task elements.
        surface_z = self.platform_size[2] / 2.0 + self.riser_height

        riser_z = (
            self.platform_size[2] / 2.0
            + self.riser_height / 2.0
        )

        wall_z = surface_z + self.wall_height / 2.0
        side_x = self.opening_width / 2.0 + side_width / 2.0

        top_z = (
            surface_z
            + self.opening_height
            + top_height / 2.0
        )

        target_z = surface_z + self.target_size[2] / 2.0

        # Main platform.
        mutils.add_shape(
            C=self.C,
            frame_name="reaching-platform",
            parent="table",
            shape=platform_shape,
            color=[0.82, 0.82, 0.82],
            joint=True,
            mass=1000.0,
        )

        # Separate riser block.
        mutils.add_shape(
            C=self.C,
            frame_name="reaching-riser",
            parent="reaching-platform",
            shape=riser_shape,
            color=[0.82, 0.82, 0.82],
            relative_position=[0.0, 0.0, riser_z],
            mass=1000.0,
        )

        # Reference frame defining the movement direction.
        self.C.addFrame(
            "reaching-direction-waypoint",
            "reaching-platform",
        )
        waypoint = self.C.getFrame("reaching-direction-waypoint")
        waypoint.setShape(ry.ST.marker, [0.01])
        waypoint.setRelativePosition([
            0.0,
            0.0,
            target_z,
        ])

        # Walls on both sides of the opening.
        for frame_name, x_position in (
            ("reaching-wall-left", -side_x),
            ("reaching-wall-right", side_x),
        ):
            mutils.add_shape(
                C=self.C,
                frame_name=frame_name,
                parent="reaching-platform",
                shape=side_shape,
                color=[0.76, 0.60, 0.42],
                relative_position=[
                    x_position,
                    0.0,
                    wall_z,
                ],
                mass=40.0,
            )

        # Wall above the opening.
        mutils.add_shape(
            C=self.C,
            frame_name="reaching-wall-top",
            parent="reaching-platform",
            shape=top_shape,
            color=[0.76, 0.60, 0.42],
            relative_position=[
                0.0,
                0.0,
                top_z,
            ],
            mass=40.0,
        )

        # Place the target object on top of the riser.
        mutils.add_shape(
            C=self.C,
            frame_name="reaching-obj",
            parent="reaching-platform",
            shape=target_shape,
            color=[0.3, 0.5, 0.8],
            relative_position=[
                0.0,
                -0.11,
                target_z,
            ],
            joint=True,
            mass=0.01,
        )

        return "reaching-platform"

    def _target_z(self):
        return (
            self.platform_size[2] / 2.0
            + self.riser_height
            + self.target_size[2] / 2.0
        )

    def getCenter(self):
        center = self.C.getFrame("reaching-platform").getPosition().copy()
        center[2] += (
            self.platform_size[2] / 2.0
            + self.riser_height
            + self.wall_height / 2.0
        )
        return np.asarray(center)

    def get_movement_direction(self):
        direction_frame = self.C.getFrame(
            "reaching-direction-waypoint"
        )

        rotation = R.from_quat(
            direction_frame.getQuaternion(),
            scalar_first=True,
        )
        direction = rotation.apply([0.0, -1.0, 0.0])

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("Invalid reaching movement direction.")

        return direction / norm

    def setTargetPosition(self):
        target = self.C.getFrame("reaching-obj")

        target_x = 0.0
        target_y = 0.0

        target.setRelativePosition([
            target_x,
            target_y,
            self._target_z()
        ])

        self.obj_pos = target.getPosition()

    def setTargetQuaternion(self):
        target = self.C.getFrame("reaching-obj")
        yaw = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
        target_quaternion = R.from_euler("z", yaw).as_quat(scalar_first=True)
        target.setRelativeQuaternion(target_quaternion)

    def update_platform_rotation(self, qua):
        """Orient the wall opening toward the robot/workspace center."""
        platform_position = self.C.getFrame("reaching-platform").getPosition()
        facing_angle = np.pi / 2.0 if platform_position[0] > 0 else -np.pi / 2.0

        placement_rotation = R.from_quat(qua, scalar_first=True)
        facing_rotation = R.from_euler("z", facing_angle)
        new_quaternion = (placement_rotation * facing_rotation).as_quat(
            scalar_first=True
        )

        self.C.getFrame("reaching-platform").setQuaternion(new_quaternion)
        return new_quaternion

    def calculate_goal_waypoint(self):
        initial_waypoint = self.C.getFrame("initial-waypoint")

        initial_position = initial_waypoint.getPosition()
        initial_quaternion = initial_waypoint.getQuaternion()

        movement_direction = self.get_movement_direction()

        # Random pushing distance in meters.
        movement_distance = np.random.uniform(0.06, 0.12)

        goal_position = (
            initial_position
            + movement_distance * movement_direction
        )
        goal_quaternion = initial_quaternion

        return goal_position, goal_quaternion

    def target_setup(self):
        target = self.C.getFrame("reaching-obj")
        center_position = target.getPosition()

        return { "reaching-obj" : [center_position]}

    def compute_task_score(
        self,
        target_history,
        max_score=0.5,
        required_distance=0.05,
        movement_threshold=0.005,
    ):
        initial_position = np.asarray(
            target_history["reaching-obj"][0],
            dtype=float,
        )

        final_position = np.asarray(
            target_history["reaching-obj"][-1],
            dtype=float,
        )

        displacement = final_position - initial_position
        displacement[2] = 0.0

        movement_distance = np.linalg.norm(displacement)

        if movement_distance < movement_threshold:
            return 0.0

        direction = np.asarray(
            self.get_movement_direction(),
            dtype=float,
        ).copy()

        direction[2] = 0.0
        direction /= np.linalg.norm(direction)

        # Projection length of the displacement onto the desired direction.
        projected_distance = np.dot(
            displacement,
            direction,
        )

        # Do not reward movement in the opposite direction.
        projected_distance = max(projected_distance, 0.0)

        task_score = max_score * np.clip(
            projected_distance / required_distance,
            0.0,
            1.0,
        )

        return float(task_score)