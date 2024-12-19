# LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors (arxiv)

### [Project Page](https://liftimage3d.github.io/) | [Arxiv Paper](https://arxiv.org/pdf/2412.09597)

[Yabo Chen](https://scholar.google.com/citations?user=6aHx1rgAAAAJ&hl=zh-TW) <sup>1*</sup>, [Chen Yang](https://scholar.google.com/citations?hl=zh-CN&user=StdXTR8AAAAJ) <sup>1*</sup>,
[Jiemin Fang](https://jaminfong.cn/) <sup>2‡</sup>, [Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN) <sup>2 </sup>,[Lingxi Xie](http://lingxixie.com/) <sup>2 </sup> </br>, [Wei Shen](https://shenwei1231.github.io/) <sup>1 </sup>,[Wenrui Dai](https://scholar.google.com/citations?user=Xg8MhyAAAAAJ&hl=en) <sup>1 </sup>, [Hongkai Xiong](https://scholar.google.com/citations?user=bB16iN4AAAAJ&hl=en&oi=ao) <sup>1 </sup>, [Qi Tian](https://www.qitian1987.com/) <sup>2 </sup>

<sup>1 </sup>Shanghai Jiao Tong University &emsp; <sup>2 </sup>Huawei Inc. &emsp;

<sup>\*</sup> equal contributions in no particular order. <sup>$\ddagger$</sup> project lead. 

![block](./imgs/teaser.png)   
Leveraging Latent Video Diffusion Models (LVDMs) priors effectively faces three key challenges: (1) degradation in quality across large camera motions, 
(2) difficulties in achieving precise camera control, and (3) geometric distortions inherent to the diffusion process that damage 3D consistency. 
We address these challenges by proposing LiftImage3D, a framework that effectively releases LVDMs' generative priors while ensuring 3D consistency. 
Specifically, we design an articulated trajectory strategy to generate video frames, which decomposes video sequences with large camera motions into ones with controllable small motions. 
Then we use robust neural matching models, i.e. MASt3R, to calibrate the camera poses of generated frames and produce corresponding point clouds. 
Finally, we propose a distortion-aware 3D Gaussian splatting representation, which can learn independent distortions between frames and output undistorted canonical Gaussians. 

![block](./imgs/method.png)
The overall pipeline of LiftImage3D. Firstly, we extend LVDM to generate diverse video clips from a single image using an
articulated camera trajectory strategy. Then all generated frames are matching using the robust neural matching module and registered in
to a point cloud. After that we initialize Gaussians from registered point clouds and construct a distortion field to model the independent
distortion of each video frame upon canonical 3DGS.

## 🦾 Updates (Still Under Construction)
- 12/13/2024: Post the arxiv paper and project page.
- 12/19/2024: Post the pipeline of LiftImage3D.

## Requirements
Pytorch 2.0 for faster training and inference.
```
conda create -f environment.yml
```
or 
```
conda create -n liftimage3d python=3.9
conda activate liftimage3d
pip install -r requirements.txt
```

Install [xformer](https://github.com/facebookresearch/xformers#installing-xformers) properly to enable efficient transformers.
```commandline
conda install xformers -c xformers
# from source
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```




##  Acknowledgement
This repository is based on original [MotionCtrl](https://github.com/TencentARC/MotionCtrl), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [4DGS](https://github.com/hustvl/4DGaussians),. Thanks for their awesome works.


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
