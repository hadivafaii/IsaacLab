import torch

from omni.isaac.lab.sensors import SensorBase, Camera, CameraCfg

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.utils import convert_dict_to_backend

def make_eyes(res) -> Camera:
    prim_utils.create_prim("/World/Head/Eye_L", "Xform")
    prim_utils.create_prim("/World/Head/Eye_R", "Xform") #FIXME: camera locations
    camera_cfg = CameraCfg(
        prim_path="/World/Head/Eye_.*/CameraSensor",
        update_period=0,
        height=res[0],
        width=res[1],
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "motion_vectors",
            "instance_segmentation_fast"
        ],
        #TODO-Theloni: make pinhole cam a fisheye cam with proper distortions
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)
    return camera

class HumanOcularSystem:
    def __init__(self, resolution) -> None:
        self.eyes = make_eyes(resolution)
        # self.eyes.set_world_poses_from_view(head_pos, look_at)


    def set_world_poses_from_view(self, head_pos, look_at):
        self.eyes.set_world_poses_from_view(head_pos, look_at) # TODO-Theloni: add torsion

    def update(self, dt):
        self.eyes.update(dt)
    
    @property
    def datadicts_np(self) -> tuple[dict, dict]:
        return (convert_dict_to_backend(self.eyes.data.output[0]), convert_dict_to_backend(self.eyes.data.output[1]))

    @property
    def data_replicator_formatted(self) -> tuple[dict, dict]:
        rep_outputs = ({"annotators": {}}, {"annotators": {}})
        for eye_idx, cam_data in enumerate(self.datadicts_np):
            cam_info = self.eyes.data.info[eye_idx]
            for key, data, info in zip(cam_data.keys(), cam_data.values(), cam_info.values()):
                if info is not None:
                    rep_outputs[eye_idx]["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_outputs[eye_idx]["annotators"][key] = {"render_product": {"data": data}}
            rep_outputs[eye_idx]["trigger_outputs"] = {"on_time": self.eyes.frame[eye_idx]}
        return rep_outputs