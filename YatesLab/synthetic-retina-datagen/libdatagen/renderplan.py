import os
import sys
import logging
from dataclasses import dataclass
from omni.isaac.kit import SimulationApp

import omni.isaac.lab.sim as sim_utils
import omni.replicator.core as rep

from .adaptivesamplepolicy import AdaptiveSamplingPolicy

class WriterProcess: # TODO-Theloni: seperate writer process
    pass

@dataclass
class RenderPlanConfig:
    write_enable = True
    outdir = "output"
    cpu = False
    throwout_start_frames = 10
    resolution = (480, 640)

class RenderPlan:
    def __init__(self, app: SimulationApp, simctx:sim_utils.SimulationContext, scene:dict, policy: AdaptiveSamplingPolicy,conf: RenderPlanConfig): # TODO: add adaptive sampling
        self.app = app
        self.sim = simctx
        self.scene = scene
        assert "eyes" in scene.keys(), "Eyes must be present in scene in order to create renderplan"
        self.policy = policy
        self.conf = conf
        
        #FIXME: not writing to correct dir
        self.lwriter =  rep.BasicWriter(
            output_dir=os.path.join(conf.outdir, "left"),
            frame_padding=0,
            colorize_instance_segmentation=True
        )
        self.rwriter =  rep.BasicWriter(
            output_dir=os.path.join(conf.outdir, "right"),
            frame_padding=0,
            colorize_instance_segmentation=True
        )

    def run(self, n_frames):
        logger = logging.getLogger()
        frame = 0
        logger.warning("######### YATESLAB: Started render plan ###########")
        while self.app.is_running() and frame < n_frames+self.conf.throwout_start_frames:
            self.sim.step()

            p = self.policy.pose()
            self.scene["eyes"].set_world_poses_from_view(*p)
            
            logger.warning(f"######### YATESLAB: rendered frame {frame} with pose {p} ###########")
            # Update camera data
            self.scene["eyes"].update(dt=self.sim.get_physics_dt())
            
            self.policy.ingest(self.scene["eyes"].eyes.data.output)
            
            
            #TODO-Theloni: update cam pose based on sampling policy
            if self.conf.write_enable and frame > self.conf.throwout_start_frames: #TODO-Theloni: merge into single write
                left, right = self.scene["eyes"].data_replicator_formatted
                self.lwriter.write(left)
                self.rwriter.write(right)

            frame += 1

    