import os
import json
import re
from argparse import ArgumentParser, Namespace



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--gpu', type=int, default=0, help="The gpu")
    parser.add_argument('--port', type=int, default=6009, help="The port")
    parser.add_argument('--cache_dir', type=str, default="output", help="The cache dictionary")
    parser.add_argument('--input_file', type=str, default='input_images/testimg_001.png', help="The extention of input images")
    

    args = parser.parse_args()
    root_dir = args.cache_dir
    # root_dir = os.path.join(args.cache_dir, os.path.basename(args.input_file).split(".")[0])
    name_list = [args.input_file.split('/')[-1].split('.')[0]]
    
    for name in name_list:
        if os.path.exists(os.path.join('output', name, 'point_cloud/iteration_13999/point_cloud.ply')):
            print("**Already down ", name)
            continue

        if not (os.path.exists(os.path.join(root_dir, name, "depth_anything_v2_align"))) :
            print("**Not exist depth_anything_v2_align", os.path.join(root_dir, name))
            continue
        if len(os.listdir(os.path.join(root_dir, name, "depth_anything_v2_align")))<200:
            print("**Not exist depth_anything_v2_align len 200", os.path.join(root_dir, name))
            continue

        if not os.path.exists(os.path.join(root_dir, name, f'input_view0.ply'))  :
            print("**Not exist input_view0.ply", os.path.join(root_dir, name))
            continue

        # print(name)
        ply_path = os.path.join(root_dir, name, "input_view0.ply")
        # print(ply_path)

        train_json = os.path.join(root_dir, name, "train_info.json")
        dict_json = [{
                "dust3r_name": f"{name}_frame0.png",
                "input_name": f"{name}_frame0.png",
                "test_name": f"{name}_frame0.png"    #For single image to 3D test to itself(will update by different test prototype)
            }]

        with open(train_json,"w+") as f:
            json.dump(dict_json,f)
            print("Finished create train_json")

        if os.path.exists(ply_path):
            try:
                cmd_str = f"CUDA_VISIBLE_DEVICES={args.gpu} python distort-3dgs/train.py -s {os.path.join(root_dir, name)}/ --port {args.port} --expname {name} --configs distort-3dgs/arguments/distort_3dgs/default.py"
                print(cmd_str)
                os.system(cmd_str)
            except Exception as e:
                print(e)
                continue