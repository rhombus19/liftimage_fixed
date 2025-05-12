Fork of https://github.com/AbrahamYabo/LiftImage3D 

![block](./assets/method.png)
The overall pipeline of LiftImage3D. Firstly, we extend LVDM to generate diverse video clips from a single image using an
articulated camera trajectory strategy. Then all generated frames are matching using the robust neural matching module and registered in
to a point cloud. After that we initialize Gaussians from registered point clouds and construct a distortion field to model the independent
distortion of each video frame upon canonical 3DGS.

## Requirements:
Cuda Toolkit (tested on cuda-12.4)
16+ GiB RAM
20+ Gib VRAM

## Setup guide
1. Clone repository.
```bash
git clone --recursive {this_repo}
cd {this_repo}
# if you have already cloned LiftImage3D:
# git submodule update --init --recursive
```

2. Install the UV Package manager
```bash
pip install uv
```

3. Install glm.hpp
```bash
sudo apt update
sudo apt install libglm-dev
```

4. Build Environment
```bash
uv sync
#NOTE: If you are using another cuda version, update the pytorch pypi index at the bottom of pyproject.toml
```

5. Download all the checkpoints needed
```bash
mkdir -p checkpoints/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt -P checkpoints/


6. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```

7. Try LiftImage3D now
```bash
uv run run_liftimg3d_motionctrl.py --cache_dir ./output --input_file input_images/1.png --width 1024 --height 768 
#Note that --width and --height need to match the actual resolution of the input image. 
#To maintain the generation performance of motionctrl, it is recommended to choose a width of 1024."
```

## Notes/Known Issues
If something breaks, you can isolate the 3 steps in our pipeline: video diffusion, mast3r, distort-3dgs in run_liftimg3d_motionctrl.py by commenting out one of the 3 lines in main()

Known issue: Unknown symbol in distort_3dgs rasterization cuda kernel


##  Acknowledgement
This repository is based on original [MotionCtrl](https://github.com/TencentARC/MotionCtrl), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [4DGS](https://github.com/hustvl/4DGaussians),. Thanks for their awesome works.


##  Citation
If you find this work repository/work helpful in your research, welcome to cite the paper and give a ⭐:

```
@misc{chen2024liftimage3d,
    title={LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors},
    author={Yabo Chen and Chen Yang and Jiemin Fang and Xiaopeng Zhang and Lingxi Xie and Wei Shen and Wenrui Dai and Hongkai Xiong and Qi Tian},
    year={2024},
    eprint={2412.09597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }
```


