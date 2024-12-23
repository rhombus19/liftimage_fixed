import os
from argparse import ArgumentParser, Namespace


def check_already_exists(file_name, image_root):
    if (os.path.exists(os.path.join(image_root,file_name+'_D')) and os.path.exists(os.path.join(image_root,file_name+'_U')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_D')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_D')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_D')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_U_frame15_U')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_U')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_U')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_Round-RI_90')) and os.path.exists(os.path.join(image_root,file_name+'_U_frame15_Round-RI_90')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_Round-RI_90')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_Round-RI_f90')) and os.path.exists(os.path.join(image_root,file_name+'_U_frame15_Round-RI_f90')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_Round-RI_f90')) ):
        print(f"**This {file_name} case has already generated all video frames with MotionCtrl-svd. skip")
        return True
    else:
        return False

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--gpu', type=int, default=0, help="The gpu names")
    # parser.add_argument('--res-dir', type=str, default="output", help="The output directions")
    parser.add_argument('--cache_dir', type=str, default="output", help="The cache dictionary")
    parser.add_argument('--height', type=int, default=576, help="The height of input images")
    parser.add_argument('--width', type=int, default=1024, help="The width of input images")
    parser.add_argument('--ddim_steps', type=int, default=50, help="The steps of DDIM")
    parser.add_argument('--input_file', type=str, default='input_images/testimg001.png', help="The extention of input images")
    parser.add_argument('--ext', type=str, default='.jpg', help="The extention of input images")
    parser.add_argument('--ckpt_motionctrl', type=str, default='checkpoints/motionctrl_svd.ckpt', help="The extention of input images")
    args = parser.parse_args()

    # ckpt='checkpoints/motionctrl_svd.ckpt'

    # if args.ext in args.input_file or not os.path.isdir(args.input_file):
    #     image_input = [args.input_file.split('/')[-1]]
    # else:
    #     image_input = [f for f in os.listdir(args.input_file) if args.ext in f]
    # cache_dir=args.cache_dir
    cache_dir = os.path.join(args.cache_dir, os.path.basename(args.input_file).split(".")[0])
    image_input = [args.input_file.split('/')[-1]]
    
    height=args.height
    width=args.width
    ckpt=args.ckpt_motionctrl

    articulated_times=2
    direction_times=4

    cond_aug=0.10
    sample_num=1

    frames=16
    seed=42
    fps=10
    motion=100


    # if not os.path.isdir(cache_dir):
    #     os.mkdir(cache_dir)
    if not os.path.isdir(os.path.join(cache_dir, 'motionctrl')):
        os.makedirs(os.path.join(cache_dir, 'motionctrl'))

    for input_img_file in image_input:
        if check_already_exists(input_img_file.split(".")[0], cache_dir):
            continue
        
        input_img = os.path.join(cache_dir, 'motionctrl', input_img_file.split(".")[0])
        if not os.path.exists(os.path.join(cache_dir, input_img_file.split('.')[0], 'motionctrl', input_img_file)):
            os.system(f"cp {args.input_file} {os.path.join(cache_dir, 'motionctrl', input_img_file)}")

        image_input_final=[]
        image_input_final.append(input_img)
        for times in range(articulated_times):
            select_input_len = len(image_input_final)
            image_input_final_this_time = image_input_final.copy()
            for image_input_final_cur in image_input_final_this_time:

                if "_Round-RI_90_frame15_Round-RI_f90" in image_input_final_cur or  "_Round-RI_f90_frame15_Round-RI_90" in image_input_final_cur or  "_U_frame15_D" in image_input_final_cur or  "_D_frame15_U" in image_input_final_cur:
                    continue
                    image_input_final.append(image_input_final_cur)

                if times > 0:
                    with_external= image_input_final_cur +".png"                
                else:
                    with_external= image_input_final_cur + args.ext


                cmd_str = f"CUDA_VISIBLE_DEVICES={args.gpu} python motionctrlsvd/main/inference/motionctrl_cmcm_run.py \
                --seed {seed} \
                --ckpt {ckpt} \
                --config 'motionctrlsvd/configs/inference/config_motionctrl_cmcm.yaml' \
                --savedir {os.path.join(cache_dir, 'motionctrl')} \
                --savefps 10 \
                --ddim_steps {args.ddim_steps} \
                --frames {frames} \
                --input {with_external} \
                --fps {fps} \
                --motion {motion} \
                --cond_aug {cond_aug} \
                --decoding_t 1 --resize \
                --height {height} --width {width} \
                --sample_num {sample_num} \
                --pose_dir 'motionctrlsvd/examples/camera_poses' \
                --transform \
                --speed 1.6"

                print(cmd_str)
                os.system(cmd_str)
                os.system(f"python motionctrlsvd/mp4topng_iterative_args.py --cur_file {image_input_final_cur} --cache_dir {cache_dir}")
                
                image_input_final.append(image_input_final_cur+"_D_frame15")
                image_input_final.append(image_input_final_cur+"_U_frame15")
                image_input_final.append(image_input_final_cur+"_Round-RI_90_frame15")
                image_input_final.append(image_input_final_cur+"_Round-RI_f90_frame15")
                image_input_final = image_input_final[1:]
