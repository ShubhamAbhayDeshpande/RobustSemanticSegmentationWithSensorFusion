"""
This script will segment a video. 

Minor changes in 'predict.py' program are made in order to adapt it to the video input. 

For now, the segmented video will not be saved anywhere. First we will only visualize the output.

The video(s) used in this script can be found in the folder '/mnt/public/media/robots/unimog-u5023/20230519_GD_bridge_first_tests/'.

"""

import argparse
import logging
import os
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

logging.basicConfig(filename = "video_input.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

def superimpose(background, foreground):    
    
    # Set adjusted colors,
    for color in range(0, 3):

        # Normalize alpha channels from 0-255 to 0-1
        alpha_background = background[:, :, color]/255.0
        alpha_foreground = foreground[:, :, color]/255.0

        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
        alpha_background * background[:, :, color] * (1-alpha_foreground)

        # set adjusted alpha and denormalize back to 0-255
        background[:, :, color] = (1-(1-alpha_foreground) * (1-alpha_background))*255

    return background


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        
        output = net(img).cpu()
        #output = net(img).device
        output = F.interpolate(output, (full_img.shape[1], full_img.shape[0]), mode="bilinear")
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument("--model", 
                        "-m", 
                        default="checkpoints/full_uni_dataset_train_30_epochs_2_batch_size_1e-5_lr/best_model/best_model.pth", 
                        type=str,
                        metavar="FILE",
                        help="Specify the file in which the model is stored"
                        )
    parser.add_argument("--input",
                        "-i",
                        default= "forest_video.mp4",
                        metavar="INPUT", 
                        type=str, 
                        help="Filenames of input images", 
                        required=False
                        )
    parser.add_argument("--output", 
                        "-o",
                        default="./checkpoints/shadow_removal_output", 
                        metavar="OUTPUT", 
                        required=False, 
                        type=str, 
                        help="Filenames of output images"
                        )
    parser.add_argument("--viz", 
                        "-v", 
                        action="store_true",
                        help="Visualize the images as they are processed")
    parser.add_argument("--save", 
                        "-s", 
                        default=True, 
                        help="Do not save the output masks"
                        )
    parser.add_argument("--mask-threshold", 
                        "-t", 
                        type=float, 
                        default=0.5,
                        help="Minimum probability value to consider a mask pixel white"
                        )
    parser.add_argument("--scale", 
                        "-sc", 
                        type=float, 
                        default=0.5,
                        help="Scale factor for the input images"
                        )
    parser.add_argument("--bilinear", 
                        action="store_true", 
                        default=False, 
                        help="Use bilinear upsampling"
                        )
    parser.add_argument("--classes", 
                        "-c", 
                        type=int, 
                        default=6, 
                        help="Number of classes"
                        )
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    # return Image.fromarray(out)
    return out

if __name__ == "__main__":

    logging.info(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")

    if ld_library_path is not None:
        logging.info(ld_library_path)
    else:
        logging.info("LD_LIBRARY_PATH is not set.")
    args = get_args()

    # Make output directory to store the masks. If directory already exists, ignore.
    os.makedirs(args.output, exist_ok=True)

    logging.info(f"started predicting masks with weights {args.model}")

    in_video = args.input

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    # 
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    # capture video from the location.
    vidcap = cv2.VideoCapture(args.input)
    success, image = vidcap.read()

    # Create an object for writing video in the location
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('./forest_superimposed.mp4', fourcc, 15, (512, 512))

    if success:
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            #print("original frame size: ", frame.shape )

            # cv2.imshow("video", frame)
            # cv2.waitKey(20)
            new_img = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = predict_img(net=net,
                               full_img=new_img,
                               scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device)
            
            out_img = mask_to_image(mask, mask_values)
            # cv2.imshow("output video", out_img)
            # cv2.waitKey(10)

            final_out = superimpose(new_img, out_img)

            out.write(final_out)

            # cv2.imshow("output_video", final_out)
            # cv2.waitKey(30)
            
            
    else:
        raise FileNotFoundError(f"Cannot load video from {args.input}")
 