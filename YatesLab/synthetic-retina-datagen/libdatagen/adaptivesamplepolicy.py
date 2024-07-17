import torch

#TODO-Theloni: implement history with replay buffers
class AdaptiveSamplingPolicy:
    pose_history: list[tuple[torch.Tensor, torch.Tensor]]
    data_history: list[torch.Tensor]
    def __init__(self, dbuf_len:int, init_camstate:tuple[torch.Tensor, torch.Tensor]):
        self.pose_history = [init_camstate]
        self.data_history = []
        self.mem_len = dbuf_len

    def ingest(self, data) -> None:
        if len(self.data_history) >= self.mem_len:
            self.data_history.pop(0)
        self.data_history.append(data)
    
    #TODO-Theloni: async pose compute with diskwriting concurrent
    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def save_posehistory(outdir:str) -> None:
        pass # TODO-Theloni: save pose path and render it to file maybe?

class RandomLookWalkPolicy(AdaptiveSamplingPolicy):
    def __init__(self, sigma: float, memory:int, init_state: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(memory, init_state)
        self.sigma = sigma

    def ingest(self, data):
        return
    
    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.sigma * torch.randn((2, 3), device=self.pose_history[-1][0].device)
        self.pose_history.append((self.pose_history[-1][0], self.pose_history[-1][1] + delta))
        return self.pose_history[-1]
    
    