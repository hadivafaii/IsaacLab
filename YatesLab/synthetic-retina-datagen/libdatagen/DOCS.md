# Documentation for libdatagen
Thelonious Cooper 2024

## `renderplan.py`
```python
@dataclass
class RenderPlanConfig:
# dataclass which holds configuration info for renderplan
    write_enable # enable file output
    outdir # absolute file path
    debug_warnings # allow debug warning outputs
    throwout_start_frames # dispose of starting frames while scene loads
    resolution # output resolution in (height, width)
```
```python
class RenderPlan:
# class that declares intent to render a sequence of frames from a scene
    def __init__(self, app: SimulationApp, 
    simctx:sim_utils.SimulationContext, 
    scene:dict, policy: 
    AdaptiveSamplingPolicy, 
    conf: RenderPlanConfig) 
    # creates new renderplan instance
    # multiple can be in play at a given time, assuming mutiple simulationcontexts are available

    def debugwarn(self, s:str) 
    # output to logger

    async def run_async(self, n_frames:int)
    # generate frames
```

## `adaptivesamplepolicy.py`
```python
class AdaptiveSamplingPolicy(abc.ABC):
# abstract base class for a policy that takes in new data and outputs new poses
    
    pose_history: list[tuple[torch.Tensor, torch.Tensor]] # cams pos (2x3), cams view target (2x3)
    data_history: list[tuple[dict, dict]]

    def __init__(self, dbuf_len:int, init_camstate:tuple[torch.Tensor, torch.Tensor])
    # construct policy class

    def ingest(self, data: tuple[dict, dict]) -> None
    # take in new data for each eye and compute the next pose

    @abc.abstractmethod
    def pose(self) -> tuple[torch.Tensor, torch.Tensor]
    # send back a pose for the camera to apply

    def save_posehistory(self, outdir:str) -> None
    # save history of poses to a numpy file
```

## `human.py`
```python
class HumanOcularSystem:
    def __init__(self, resolution) -> None
    # instantiate cameras

    def set_world_poses_from_view(self, head_pos, look_at)
    # set poses

    def update(self, dt)
    # apply physics step

    @property
    def datadicts_np(self) -> tuple[dict, dict]
    # export data to numpy format 

    @property
    def data_replicator_formatted(self) -> tuple[dict, dict]
    # export data to omniverse replicator format
```
## `retinasensor.py`
```python
class RecurrentConeModel(nn.Module):
# pytorch model which implements 1st order eulers method for solving rieke's cone model

    def __init__(self, foveal = True, dt = 0.001)
    # instantate class with parameters for foveal vs peripheral cones

    def init_parameters(self, stim:torch.Tensor)
    # first pass

    def forward(self, stim:torch.Tensor)
    # subsequent passes

class FovealConeSensor(Camera):
# sensor class inheriting from camera that also generates cone signal

    def __init__(self, **kwargs)
    # instantate class based on cfg=CameraCfg()
    
    def _update_buffers_impl(self, env_ids: Sequence[int])
    # generate foveal cone data from rgb output
```
## `sceneload.py`
```python
def load_scene(fpath: str)
# load a scene from a .usd file

def random_forest()
# randomly generated forest scene with trees, rocks, random time of day, and agents
# currently unimplemented

def default_scene()
# default scene with some random primitives dropping onto a ground plane
```