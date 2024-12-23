import cv2
import os
from argparse import ArgumentParser, Namespace

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--cur_file', type=str, default="", help="The input file names")
    parser.add_argument('--cache_dir', type=str, default="", help="The res file names") 
    args = parser.parse_args()

    root_dir = output_dir = os.path.join(args.cache_dir, 'motionctrl')

    list_dir = ['_Round-RI_90',
                '_Round-RI_f90',
                '_D',
                '_U',
                ]

    for idx, dir_name in enumerate(list_dir):
        item = args.cur_file.split("/")[-1] + dir_name + ".mp4"
        # print("mp4topng_iterative ", item)
        vidcap = cv2.VideoCapture(os.path.join(root_dir, item))
        if not os.path.isdir(os.path.join(output_dir, item.replace(".mp4",""))):
            os.mkdir(os.path.join(output_dir, item.replace(".mp4","")))
        success,image = vidcap.read()
        count = 0
        print(str(idx) + " " + item)
        while success:
            cv2.imwrite(os.path.join(output_dir,item.replace(".mp4",""),item.replace(".mp4","")+"_frame%d.png"%count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1
        
        cmd = "cp " + os.path.join(output_dir,item.replace(".mp4",""),item.replace(".mp4","")+"_frame15.png") + " " + os.path.join(output_dir,item.replace(".mp4","")+"_frame15.png")
        print(cmd)
        os.system(cmd)

