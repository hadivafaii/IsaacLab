from dataclasses import dataclass

import omni.isaac.lab.sim as sim_utils

from .human import HumanOcularSystemCfg
class WriterProcess: # TODO-Theloni: seperate writer process
    pass

@dataclass
class RenderPlanConfig:
    write_enable = True
    outdir = "./output"
    cpu = False
    throwout_start_frames = 5
    resolution = (480, 640)
    annotations = ["rgb", "distance_to_image_plane", "instance_segmentation_fast"]
    ocular_settings = HumanOcularSystemCfg() # get ocsyssensorcfg

class RenderPlan:
    def __init__(self, simctx:sim_utils.SimulationContext, scene:dict, conf: RenderPlanConfig):
        self.sim = simctx
        self.scene = scene
        self.conf = conf
    
    def run(n_frames):
        #WRITEME-Theloni: sim runner
        pass
    