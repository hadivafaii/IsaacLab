import os
import asyncio
import nest_asyncio
import logging
from dataclasses import dataclass

from omni.isaac.kit import SimulationApp

import omni.isaac.lab.sim as sim_utils

from .adaptivesamplepolicy import AdaptiveSamplingPolicy

async def write_renderframe(writedir: str, queue: asyncio.Queue):
    tdict = await queue.get()
    try:
        os.makedirs(writedir)
    except FileExistsError:
        pass
    td_cpu = tdict.cpu()

    td_cpu.memmap_(writedir, num_threads=32)
    del tdict
    del td_cpu
    queue.task_done()

@dataclass
class RenderPlanConfig:
    write_enable: bool = True
    outdir: str = "output"
    debug_warnings: bool = False
    throwout_start_frames: int = 10
    resolution: tuple[int, int] = (480, 640)

class RenderPlan:
    def __init__(self, app: SimulationApp, simctx:sim_utils.SimulationContext, scene:dict, policy: AdaptiveSamplingPolicy, conf: RenderPlanConfig):
        self.app = app
        self.sim = simctx
        self.scene = scene
        assert "eyes" in scene.keys(), "Eyes must be present in scene in order to create renderplan"
        self.policy = policy
        self.conf = conf
        self.logger = logging.getLogger()

        self.leftwq = asyncio.Queue()
        self.rightwq = asyncio.Queue()
        self.writetasks = []

    def debugwarn(self, s:str):
        if self.conf.debug_warnings:
            self.logger.warning(s)

    async def run_async(self, n_frames:int):
        try:
            eloop = asyncio.get_event_loop()
            nest_asyncio.apply(eloop)
        except RuntimeError:
            eloop = asyncio.new_event_loop()
            nest_asyncio.apply(eloop)

        frame = 0
        self.debugwarn("######### YATESLAB: Started render plan ###########")
        while self.app.is_running() and frame < n_frames+self.conf.throwout_start_frames:
            self.sim.step()
            # logger.warning("######### YATESLAB STEPTIME ##########")
            p = self.policy.pose()
            self.scene["eyes"].set_world_poses_from_view(*p)

            self.debugwarn(f"######### YATESLAB: rendered frame {frame - self.conf.throwout_start_frames} ###########")
            # Update camera data
            self.scene["eyes"].update(dt=self.sim.get_physics_dt())

            self.policy.ingest(self.scene["eyes"].eyes.data.output)

            if self.conf.write_enable and frame > self.conf.throwout_start_frames: #TODO-YatesLab merge into single write
                left, right = self.scene["eyes"].eyes.data.output
                self.leftwq.put_nowait(left)
                self.rightwq.put_nowait(right)
                self.writetasks.append(eloop.create_task(write_renderframe(os.path.join(self.conf.outdir,"left", f"{frame - self.conf.throwout_start_frames}"), self.leftwq)))
                self.writetasks.append(eloop.create_task(write_renderframe(os.path.join(self.conf.outdir,"right", f"{frame - self.conf.throwout_start_frames}"), self.rightwq)))

            frame += 1

        await asyncio.gather(*self.writetasks, return_exceptions=True)
