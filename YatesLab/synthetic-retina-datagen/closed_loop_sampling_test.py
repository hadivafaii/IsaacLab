###### IMPORTS ######
import argparse

from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script renders a scene according to an adaptive sampling policy")
parser.add_argument("--nframes", default=200)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Can only import omni classes after launching the omniverse backend
import omni.isaac.lab.sim as sim_utils

# necessary for async file writing
import asyncio
import nest_asyncio
nest_asyncio.apply()

import torch
torch.set_grad_enabled(False) # cut out memory leak from gradient tracking

# import necessary project library content
from libdatagen.renderplan import RenderPlan, RenderPlanConfig
from libdatagen.adaptivesamplepolicy import StaticPolicy, RandomLookWalkPolicy
from libdatagen.human import HumanOcularSystem
from libdatagen.sceneload import default_scene, load_scene

###### SCRIPT STARTS HERE ######
import carb
print(carb.settings.get_settings())

render_conf = RenderPlanConfig()
render_conf.debug_warnings = True
render_conf.resolution = (960, 1280)
render_conf.outdir = "/home/theloni/IsaacLab/YatesLab/synthetic-retina-datagen/output" # must be absolute filepath

sim_cfg = sim_utils.SimulationCfg(device="cuda", dt=1/100) # device must be "cuda"
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0)) # sets default viewport camera pose, will not be used in actual rendering

scene =   default_scene()
scene["eyes"] = HumanOcularSystem(render_conf.resolution) # must create "eyes" element for rendering
sim.reset() # inform the simulation ctx of the scene update

# create policy for pose updates
cam_position = torch.tensor([[2.5, 2.5, 2.5], [2.51, 2.5, 2.5]], device=sim.device)
cam_target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
policy = StaticPolicy((cam_position, cam_target)) #RandomLookWalkPolicy(0.01, -1, (cam_position, cam_target))

# instantiate renderplan
plan = RenderPlan(simulation_app, sim, 
                  scene, policy, 
                  render_conf)

# generate frames
asyncio.run(plan.run_async(args_cli.nframes))

simulation_app.close() # clean up background processes