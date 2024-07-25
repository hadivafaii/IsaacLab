from typing import Sequence

import torch
import torch.nn as nn

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.lab.sensors import Camera

class RecurrentConeModel(nn.Module):
    def __init__(self, foveal = True, dt = 0.001):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(0.02,dtype=torch.float32))
        self.h = nn.Parameter(torch.tensor(3.,dtype=torch.float32))
        self.m = nn.Parameter(torch.tensor(4.,dtype=torch.float32))
        self.q = nn.Parameter(torch.tensor(0.1125,dtype=torch.float32))
        self.Ca_dark = nn.Parameter(torch.tensor(1.,dtype=torch.float32))
        self.I_dark = nn.Parameter(torch.tensor(80.,dtype=torch.float32))
        self.G_dark = nn.Parameter(torch.tensor(20.,dtype=torch.float32))
        self.Gamma = nn.Parameter(torch.tensor(10.,dtype=torch.float32))
        self.Sigma = nn.Parameter(torch.tensor(10. if foveal else 22.,dtype=torch.float32))      #22 peripheral, 10 foveal
        self.Phi = nn.Parameter(torch.tensor(22.,dtype=torch.float32))
        self.Beta = nn.Parameter(torch.tensor(9.,dtype=torch.float32))
        self.Beta_slow = nn.Parameter(torch.tensor(0.4,dtype=torch.float32))
        self.eta = nn.Parameter(torch.tensor(700. if foveal else 2000.,dtype=torch.float32))       #2000 peripheral, 700 foveal
        self.Kgc = nn.Parameter(torch.tensor(0.5,dtype=torch.float32))
        self.dt = torch.tensor(dt)

    def init_parameters(self, stim):
        self.R_prev = self.Gamma * stim / self.Sigma # was stim[0]
        self.P_prev = (self.R_prev + self.eta) / self.Phi
        self.Ca_prev = self.I_dark * self.q  / self.Beta
        self.Ca_slow_prev = self.Ca_prev
        self.kCa_prev = self.k * (1/(1+(self.Ca_slow_prev/self.Ca_dark)))

        self.G_prev = (self.I_dark/self.kCa_prev) ** (1/self.h)
        self.Smax = self.P_prev * self.G_prev * (1 + (self.Ca_prev/self.Kgc)**self.m)
        self.S_prev = self.Smax / (1 + (self.Ca_prev/self.Kgc)**self.m)

        self.G_prev = (self.S_prev/self.P_prev)
        self.I_prev = self.kCa_prev * self.G_prev**self.h


    def forward(self, stim:torch.Tensor):
        """
        Input [W, H, C]
        Output [W, H, C]
        """

        R_curr = self.R_prev + self.dt*(self.Gamma*stim - self.Sigma*self.R_prev)
        P_curr = self.P_prev + self.dt*(self.R_prev + self.eta - self.Phi*self.P_prev)
        Ca_curr = self.Ca_prev + self.dt*(self.q*self.I_prev - self.Beta*self.Ca_prev)
        Ca_slow_curr = self.Ca_slow_prev + self.dt*(self.Beta_slow * (self.Ca_slow_prev - self.Ca_prev))
        kCa_curr = self.k * (1 / (1+(self.Ca_slow_prev/self.Ca_dark)))
        G_curr = self.G_prev + self.dt*(self.S_prev - self.P_prev*self.G_prev)
        S_curr = self.Smax / (1 + (Ca_curr/self.Kgc)**self.m)

        I_curr = (kCa_curr * G_curr**self.h)

        #Update values
        self.R_prev = R_curr
        self.P_prev = P_curr
        self.Ca_prev = Ca_curr
        self.Ca_slow_prev = Ca_slow_curr
        self.kCa_prev = kCa_curr
        self.G_prev = G_curr
        self.S_prev = S_curr
        self.I_prev = I_curr

        return I_curr


class FovealConeSensor(Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        s  = SimulationContext.instance()
        # self.lumweights = torch.tensor([0.2126, 0.7152, 0.0722], device=s.device).view((1,1,3))
        self.cones = (RecurrentConeModel(foveal=True, dt = s.get_physics_dt()).to(s.device), 
                      RecurrentConeModel(foveal=True, dt = s.get_physics_dt()).to(s.device))
    
    #TODO: channel-dependent blur to sim chromatic abberation
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1
        # -- pose
        self._update_poses(env_ids)
        # -- read the data from annotator registry
        # check if buffer is called for the first time. If so then, allocate the memory
        if len(self._data.output.sorted_keys) == 0:
            # this is the first time buffer is called
            # it allocates memory for all the sensors
            self._create_annotator_data()
            for index in env_ids:
                self.cones[index].init_parameters(torch.zeros((self.cfg.height, self.cfg.width, 3), device=self.device, dtype=torch.float32))
        else:
            # iterate over all the data types
            for name, annotators in self._rep_registry.items():
                # iterate over all the annotators
                for index in env_ids:
                    # get the output
                    output = annotators[index].get_data()
                    # process the output
                    data, info = self._process_annotator_output(name, output)
                    # add data to output
                    self._data.output[name][index] = data
                    # add info to output
                    self._data.info[index][name] = info
                    if name == "rgb":
                        if "cones" not in self._data.output.keys():
                            self._data.output["cones"] = torch.zeros((2, self.cfg.height, self.cfg.width, 3), dtype=torch.float32, device=self.device)

                        self._data.output["cones"][index] = self.cones[index](data.to(dtype=torch.float32)[:, :, :-1]) # no alpha channel
                        #TODO-YatesLab scale luminance data to Rhodopsin rates and add poisson noise in accordance with Rieke paper
