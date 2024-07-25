import os
import argparse
parser = argparse.ArgumentParser(description="This script renders memmap tensor data to mp4 videos for easy viewing")
parser.add_argument("--out", default="/home/theloni/IsaacLab/YatesLab/synthetic-retina-datagen/output")
parser.add_argument("--frames", default=180)
parser.add_argument("--fps", default=60)
args = parser.parse_args()

import torch
from tensordict import TensorDict
import torchvision.io as tvio

ELEMENTS = ["rgb", "distance_to_image_plane", "normals", "motion_vectors", "instance_segmentation_fast", "cones"]

container_l = TensorDict({}, batch_size=args.frames)
container_r = TensorDict({}, batch_size=args.frames)

for i in range(1, args.frames):
    container_l[i] = TensorDict.load_memmap(os.path.join(args.out, "left", f"{i}"))
    container_r[i] = TensorDict.load_memmap(os.path.join(args.out, "right", f"{i}"))

for key in ELEMENTS:
    l = container_l[key].clone().detach()
    r = container_r[key].clone().detach()
    print(key, "type", l.dtype, "shape", l.shape, "minmax", l.min(), l.max())

    if l.shape[-1] == 4: # remove alpha channel
        l = l[:,:,:, :-1]
        r = r[:,:,:, :-1]
    
    if l.dtype == torch.float32: # cast to ints
        l += l.min()
        l /= l.max()
        l *= 255
        l = l.to(dtype=torch.uint8)
        if l.dim() == 3: # if its 1 channel make it 
            l = torch.repeat_interleave(l.unsqueeze(-1), 3, dim = -1)
        
        r += r.min()
        r /= r.max()
        r *= 255
        r = r.to(dtype=torch.uint8)
        if r.dim() == 3:
            r = torch.repeat_interleave(r.unsqueeze(-1), 3, dim = -1)

    tvio.write_video(os.path.join(args.out, f"{key}_left.mp4"), l, fps=args.fps)
    tvio.write_video(os.path.join(args.out, f"{key}_right.mp4"), r, fps=args.fps)