import os
from argparse import ArgumentParser, Namespace

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # parser.add_argument('--num', type=int, default=0, help="The input file names")
    parser.add_argument('--gpu', type=int, default=0, help="The gpu names")
    parser.add_argument('--cache_dir', type=str, default="output", help="The cache dictionary")
    parser.add_argument('--height', type=int, default=576, help="The height of input images")
    parser.add_argument('--width', type=int, default=1024, help="The width of input images")
    parser.add_argument('--ddim_steps', type=int, default=50, help="The steps of DDIM")
    parser.add_argument('--input_file', type=str, default='input_images/testimg001.png', help="The extention of input images")
    # parser.add_argument('--ext', type=str, default='.png', help="The extention type of input images")
    parser.add_argument('--ckpt_motionctrl', type=str, default='checkpoints/motionctrl_svd.ckpt', help="The ckpt path of motionctrl")
    parser.add_argument('--ckpt_mast3r', type=str, default='checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="The ckpt path of mast3r")
    parser.add_argument('--ckpt_depthanythingv2', type=str, default='checkpoints/depth_anything_v2_vitl.pth', help="The depth anything v2 ckpt")
    parser.add_argument('--port', type=int, default=6009, help="The port of program")

    args = parser.parse_args()
    
    _, ext = os.path.splitext(args.input_file)
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)
    cmd_motionctrl = f"python motionctrlsvd/configs/inference/scripts_motionctrl.py --gpu {args.gpu} \
                                                                        --cache_dir {args.cache_dir} \
                                                                        --height {args.height} \
                                                                        --width {args.width} \
                                                                        --input_file {args.input_file} \
                                                                        --ext {ext} \
                                                                        --ckpt_motionctrl {args.ckpt_motionctrl}"
    cmd_mast3r = f"python mast3r/calc_pose_inthewild_motionctrl.py  --gpu {args.gpu} \
                                                                    --cache_dir {args.cache_dir} \
                                                                    --height {args.height} \
                                                                    --width {args.width} \
                                                                    --input_file {args.input_file} \
                                                                    --ext {ext} \
                                                                    --ckpt_mast3r {args.ckpt_mast3r} \
                                                                    --ckpt_depthanythingv2 {args.ckpt_depthanythingv2}"
    
    cmd_distort3dgs = f"python distort-3dgs/cmd_3dgs_motionctrl_inthewild.py  --gpu {args.gpu} \
                                                                    --cache_dir {args.cache_dir} \
                                                                    --input_file {args.input_file} \
                                                                    --port {args.port} "
                                                                    
    
    os.system(cmd_motionctrl)
    os.system(cmd_mast3r)
    os.system(cmd_distort3dgs)
    
