import random
import asyncio
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip

# import omni.simready.explorer as sre # not yet supported
from pxr import Gf, Sdf, Usd


import omni.replicator.core as rep

def load_scene(fpath):
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    scene_entities = {}
    
    map_file_cfg = sim_utils.UsdFileCfg(usd_path=fpath)
    
    scene_entities[f"map"] = map_file_cfg.func("/World/Map", map_file_cfg)
    return scene_entities
    
#WRITEME-YatesLab random forest with trees, rocks, random time of day, and agents
# ref for random time of day: https://forums.developer.nvidia.com/t/randomize-time-of-day-in-dynamic-sky/273833/5
def random_forest():
    # Load the sky
    rep.create.from_usd(usd="https://omniverse-content-production.s3.us-west-2.amazonaws.com/Environments/2023_1/DomeLights/Dynamic/CumulusLight.usd")

    # Select the objects in the stage
    sky = rep.get.prim_at_path(path="/Replicator/Ref_Xform/Ref/Looks/SkyMaterial/Shader")

    # Set sun position and time of day
    with sky:
        rep.modify.attribute(name="inputs:SunPositionFromTOD", value=True)
        rep.modify.attribute(name="inputs:TimeOfDay",value=rep.distribution.uniform(lower=0.0,upper=23.5))

    # Parse terrain generation
    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=False, color_scheme="none")

    # Handler for terrains importing
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=1,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )

    # Create terrain importer
    terrain_importer = TerrainImporter(terrain_importer_cfg)
    
    scene_entities = {"terrain": terrain_importer, "sky": sky, "surroundings" : []}

    assets = asyncio.run(sre.find_assets(search_words=["tree", "shrub", "bush", "rock", "boulder"]))
    print(f"Found {len(assets)} assets")

    # 2. Prepare to configure the assets
    # All SimReady Assets have a Physics behavior, which is implemented as a
    # variantset named PhysicsVariant. To enable rigid body physics on an asset,
    # this variantset needs to be set to "RigidBody".
    variants = {"PhysicsVariant": "RigidBody"}

    # 3. Add all assets found in step (1) to the current stage as a payload

    for i, asset in enumerate(assets):
        pos = -200 + 200 * i
        res, prim_path = sre.add_asset_to_stage(
            asset.main_url, position=Gf.Vec3d(pos, 0, -pos), variants=variants, payload=True
        )
        if res:
            scene_entities["surroundings"].append(res)
            print(f"Added '{prim_path}' from '{asset.main_url}'")

    
    return scene_entities

def default_scene():
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Xform to hold objects
    prim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(8):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)
        # Rigid Object
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)
    return scene_entities