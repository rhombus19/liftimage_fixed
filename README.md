# LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors (arxiv)

### [Project Page](https://liftimage3d.github.io/) | [Arxiv Paper](https://arxiv.org/pdf/2412.09597)

[Yabo Chen](https://scholar.google.com/citations?user=6aHx1rgAAAAJ&hl=zh-TW) <sup>1*</sup>, [Chen Yang](https://scholar.google.com/citations?hl=zh-CN&user=StdXTR8AAAAJ) <sup>1*</sup>,
[Jiemin Fang](https://jaminfong.cn/) <sup>2‡</sup>, [Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN) <sup>2 </sup>,[Lingxi Xie](http://lingxixie.com/) <sup>2 </sup> </br>, [Wei Shen](https://shenwei1231.github.io/) <sup>1 </sup>,[Wenrui Dai](https://scholar.google.com/citations?user=Xg8MhyAAAAAJ&hl=en) <sup>1 </sup>, [Hongkai Xiong](https://scholar.google.com/citations?user=bB16iN4AAAAJ&hl=en&oi=ao) <sup>1 </sup>, [Qi Tian](https://www.qitian1987.com/) <sup>2 </sup>

<sup>1 </sup>Shanghai Jiao Tong University &emsp; <sup>2 </sup>Huawei Inc. &emsp;

<sup>\*</sup> equal contributions in no particular order. <sup>$\ddagger$</sup> project lead. 

![block](./assets/teaser.png)   
Leveraging Latent Video Diffusion Models (LVDMs) priors effectively faces three key challenges: (1) degradation in quality across large camera motions, 
(2) difficulties in achieving precise camera control, and (3) geometric distortions inherent to the diffusion process that damage 3D consistency. 
We address these challenges by proposing LiftImage3D, a framework that effectively releases LVDMs' generative priors while ensuring 3D consistency. 
Specifically, we design an articulated trajectory strategy to generate video frames, which decomposes video sequences with large camera motions into ones with controllable small motions. 
Then we use robust neural matching models, i.e. MASt3R, to calibrate the camera poses of generated frames and produce corresponding point clouds. 
Finally, we propose a distortion-aware 3D Gaussian splatting representation, which can learn independent distortions between frames and output undistorted canonical Gaussians. 

![block](./assets/method.png)
The overall pipeline of LiftImage3D. Firstly, we extend LVDM to generate diverse video clips from a single image using an
articulated camera trajectory strategy. Then all generated frames are matching using the robust neural matching module and registered in
to a point cloud. After that we initialize Gaussians from registered point clouds and construct a distortion field to model the independent
distortion of each video frame upon canonical 3DGS.

## 🦾 Updates (Still Under Construction)
- 12/13/2024: Post the arxiv paper and project page.
- 12/23/2024: Post the pipeline of LiftImage3D and requirements.

## Requirements
1. Clone LiftImage3D.
```bash
git clone --recursive https://github.com/AbrahamYabo/LiftImage3D
cd LiftImage3D
# if you have already cloned LiftImage3D:
# git submodule update --init --recursive
```

2. Pytorch 2.0 for faster training and inference.
```bash
conda create -n liftimage3d python=3.11
conda activate liftimage3d
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system

pip install -r requirements.txt
```

3. Install [xformer](https://github.com/facebookresearch/xformers#installing-xformers) properly to enable efficient transformers.
```bash
conda install xformers -c xformers
# from source
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

5. Install the submodules of 3DGS. We have two diff-gaussian-rasterization for different rendering strategy. [diff-w](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/tree/43e21bff91cd24986ee3dd52fe0bb06952e50ec7) and [diff-ori](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0) 
```bash
# We have two diff-gaussian-rasterization for different rendering strategy
cd distort-3dgs
pip install -e submodules/diff-w
pip install -e submodules/diff-ori
pip install -e submodules/simple-knn
cd ../
```

6. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```

7. Download all the checkpoints needed
```bash
mkdir -p checkpoints/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt -P checkpoints/
```
Because the computing resources I have cannot directly access the web network, I choose to keep laion/CLIP-ViT-H-14-laion2B-s32B-b79K locally. You can also download it from the website https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main for access.
All the checkpoints should be organize as follows.
```
├── checkpoints
│     ├── depth_anything_v2_vitl.pth
│     ├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
│     ├── motionctrl_svd.ckpt
├── laion
│   | CLIP-ViT-H-14-laion2B-s32B-b79K
│     ├── config.json
│     ├── open_clip_config.json
│     ├── open_clip_pytorch_model.bin
│     ├── ...
```

8. Try LiftImage3D now
```bash
python run_liftimg3d_motionctrl.py --cache_dir ./output --input_file input_images/testimg001.png --width 1024 --height 768 
#Note that --width and --height need to match the actual resolution of the input image. 
#To maintain the generation performance of motionctrl, it is recommended to choose a width of 1024."
```

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

##  On Coming
- Release the code based on ViewCrafter
- Release the code of test prototype
- Release the local 3DGS viewer
