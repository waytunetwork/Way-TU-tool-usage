import random
import robotic as ry
import numpy as np

import WayTu_RAI.model_utils as mutils
from scipy.spatial.transform import Rotation as R


class GenerateAdditionalTools: 
    def __init__(self, cfg, C):
        self.C = C
        self.cfg = cfg 

    def getTools(self):
        objs = [
            self.create_pitcher,
        ]
        return objs


    def create_U_tool(self):
        bridge_shape = [0.16, 0.035, 0.02, 0.004]
        arm_shape = [0.035, 0.20, 0.02, 0.004]

        arm_offset_x = (bridge_shape[0] - arm_shape[0]) / 2.0
        arm_offset_y = (arm_shape[1] - bridge_shape[1]) / 2.0

        color = [0.45, 0.55, 0.75]

        mutils.add_shape(
            C=self.C,
            frame_name="U-tool-base",
            parent="table",
            shape=bridge_shape,
            joint=True,
            mass=0.05,
            color=color,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="U-tool-head",
            parent="U-tool-base",
            shape=arm_shape,
            relative_position=[-arm_offset_x, arm_offset_y, 0.0],
            mass=0.025,
            color=color,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="U-tool-head2",
            parent="U-tool-base",
            shape=arm_shape,
            relative_position=[arm_offset_x, arm_offset_y, 0.0],
            mass=0.025,
            color=color,
        )

        return "L-ruler"
    
    def create_pitcher(self):
        body_width, body_depth, body_height = 0.10, 0.09, 0.12
        bottom_thickness, wall_thickness = 0.012, 0.008

        pitcher_color = [0.92, 0.58, 0.20]
        handle_color = [0.30, 0.22, 0.15]

        # The bottom is the root of the complete pitcher.
        mutils.add_shape(
            C=self.C, frame_name="pitcher-base", parent="table",
            shape=[body_width, body_depth, bottom_thickness, 0.004,],
            joint=True,
            color=pitcher_color,
        )

        wall_z = bottom_thickness / 2.0 + body_height / 2.0

        # Side walls.
        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-wall-left",
            parent="pitcher-base",
            shape=[
                wall_thickness,
                body_depth,
                body_height,
                0.003,
            ],
            relative_position=[
                -body_width / 2.0 + wall_thickness / 2.0,
                0.0,
                wall_z,
            ],
            color=pitcher_color,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-wall-right",
            parent="pitcher-base",
            shape=[
                wall_thickness,
                body_depth,
                body_height,
                0.003,
            ],
            relative_position=[
                body_width / 2.0 - wall_thickness / 2.0,
                0.0,
                wall_z,
            ],
            color=pitcher_color,
        )

        # Back and front walls.
        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-wall-back",
            parent="pitcher-base",
            shape=[
                body_width,
                wall_thickness,
                body_height,
                0.003,
            ],
            relative_position=[
                0.0,
                -body_depth / 2.0 + wall_thickness / 2.0,
                wall_z,
            ],
            color=pitcher_color,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-wall-front",
            parent="pitcher-base",
            shape=[
                body_width,
                wall_thickness,
                body_height,
                0.003,
            ],
            relative_position=[
                0.0,
                body_depth / 2.0 - wall_thickness / 2.0,
                wall_z,
            ],
            color=pitcher_color,
        )

        # Spout extending in the positive y-direction.
        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-spout",
            parent="pitcher-base",
            shape=[0.042, 0.075, 0.016, 0.003],
            relative_position=[0.0, 0.078, 0.105],
            color=pitcher_color,
        )

        # Handle connectors on the negative y-side.
        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-handle-lower",
            parent="pitcher-base",
            shape=[0.026, 0.060, 0.016, 0.004],
            relative_position=[0.0, -0.070, 0.040],
            color=handle_color,
        )

        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-handle-upper",
            parent="pitcher-base",
            shape=[0.026, 0.060, 0.016, 0.004],
            relative_position=[0.0, -0.070, 0.100],
            color=handle_color,
        )

        # Graspable vertical part of the handle.
        mutils.add_shape(
            C=self.C,
            frame_name="pitcher-handle-grip",
            parent="pitcher-base",
            shape=[0.028, 0.018, 0.080, 0.005],
            relative_position=[0.0, -0.098, 0.070],
            color=handle_color,
        )

        return "pitcher"
    
    def forked_paddle(self):
        handle_shape = [0.025, 0.25, 0.025, 0.005]

        # Short horizontal connector between the handle and prongs
        connector_shape = [0.09, 0.02, 0.01, 0.005]

        # Each prong is long and narrow
        prong_shape = [0.018, 0.08, 0.01, 0.004]

        random_handle_shape = self.add_shape_randomness(handle_shape)
        random_connector_shape = self.add_shape_randomness(connector_shape)
        random_prong_shape = self.add_shape_randomness(prong_shape)

        connector_position = [0.0, 0.09, 0.0]
        connector_position, connector_qua = self.add_head_randomness(
            connector_position
        )

        mutils.add_shape(
            C=self.C,
            frame_name="spatula-base",
            parent="table",
            shape=random_handle_shape,
            joint=True,
            color=[0.15, 0.15, 0.15],
        )

        mutils.add_shape(
            C=self.C,
            frame_name="spatula-head",
            parent="spatula-base",
            shape=random_connector_shape,
            relative_position=connector_position,
            relative_quaternion=connector_qua,
            color=[0.8, 0.8, 0.8],
        )

        prong_offsets = [-0.032, 0.0, 0.032]

        for index, offset_x in enumerate(prong_offsets):
            mutils.add_shape(
                C=self.C,
                frame_name=f"spatula-prong-{index}",
                parent="spatula-head",
                shape=random_prong_shape,
                relative_position=[offset_x, 0.045, 0.0],
                color=[0.8, 0.8, 0.8],
            )

        self.last_variant = "forked-paddle"

        # Canonical family expected by the existing pipeline
        return "spatula"