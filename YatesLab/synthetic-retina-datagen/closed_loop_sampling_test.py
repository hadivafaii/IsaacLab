import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script renders a scene according to an adaptive sampling policy")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils

import torch 

from libdatagen.renderplan import RenderPlan, RenderPlanConfig
from libdatagen.adaptivesamplepolicy import RandomLookWalkPolicy
from libdatagen.human import HumanOcularSystem
from libdatagen.sceneload import default_scene

render_conf = RenderPlanConfig()
render_conf.outdir = "~/IsaacLab/YatesLab/synthetic-retina-datagen/output"

sim_cfg = sim_utils.SimulationCfg(device="cuda") # device must be "cuda"
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

scene = default_scene()
scene["eyes"] = HumanOcularSystem(render_conf.resolution) # must create "eyes" element
sim.reset() # inform the simulation ctx of the scene update


cam_position = torch.tensor([[2.5, 2.5, 2.5], [2.51, 2.5, 2.5]], device=sim.device)
cam_target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)

# instantiate renderplan
plan = RenderPlan(simulation_app, sim, 
                  scene, RandomLookWalkPolicy(0.05, -1, (cam_position, cam_target)), 
                  render_conf)

plan.run(50) # generate 50 frames

simulation_app.close()
