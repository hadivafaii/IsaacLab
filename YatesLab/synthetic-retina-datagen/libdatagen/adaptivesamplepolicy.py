import abc
import numpy as np
import torch

#TODO-YatesLab implement history with replay buffers
class AdaptiveSamplingPolicy(abc.ABC):
    pose_history: list[tuple[torch.Tensor, torch.Tensor]] # cams pos (2x3), cams view target (2x3)
    data_history: list[tuple[dict, dict]]

    def __init__(self, dbuf_len:int, init_camstate:tuple[torch.Tensor, torch.Tensor]):
        self.pose_history = [init_camstate]
        self.data_history = []
        self.mem_len = dbuf_len

    def ingest(self, data: tuple[dict, dict]) -> None:
        if len(self.data_history) >= self.mem_len:
            self.data_history.pop(0)
        self.data_history.append(data)

    @abc.abstractmethod
    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def save_posehistory(self, outdir:str) -> None:
        stacked = torch.stack(tuple(torch.stack((pos, targets)) for pos, targets in self.pose_history))
        np.save(outdir, stacked.numpy())

class RandomLookWalkPolicy(AdaptiveSamplingPolicy):
    def __init__(self, sigma: float, memory:int, init_state: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(memory, init_state)
        self.sigma = sigma

    def ingest(self, data):
        return

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.sigma * torch.randn((1, 3), device=self.pose_history[-1][0].device).repeat((2, 1))
        self.pose_history.append((self.pose_history[-1][0], self.pose_history[-1][1] + delta))
        return self.pose_history[-1]
    
    