from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset): #Follow the setting of 4DGS to package Camera twice.
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()
        if self.dataset_type == "dust3r":
            caminfo = self.dataset[index]
            image = caminfo.image
            image_name = caminfo.image_name
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            mask = caminfo.mask
            depth = caminfo.mono_depth

            orig_w, orig_h = caminfo.image.size
            resolution = (int(orig_w), int(orig_h))

            resized_image_rgb = PILtoTorch(caminfo.image, resolution)

            gt_image = resized_image_rgb[:3, ...]

            loaded_mask = None

            if resized_image_rgb.shape[1] == 4:
                loaded_mask = resized_image_rgb[3:4, ...]

            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=gt_image,gt_alpha_mask=loaded_mask,
                              image_name=image_name,uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask,depth=depth)
        
        elif self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
