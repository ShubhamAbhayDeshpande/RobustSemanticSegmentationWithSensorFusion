"""
This is the script for semantic segmentation with shadow removal images. The model used is trained on regular rgb images. 

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
import pdb
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

logging.basicConfig(filename = "predict_shadow_removal_images_with_time.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

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
                        default= "samples_shadow_removal",
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

    return Image.fromarray(out)


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

    in_files = args.input
    # Decide whether to store the output images in the given path for o/p folder or in the same folder as i/p folder.
    out_files = get_output_filenames(args)  

    # ld_library_path = os.environ.get("LD_LIBRARY_PATH")

    # if ld_library_path is not None:
    #     pass
    # else:
    #     raise ValueError("The path to the variable LD_LIBRARY_PATH is not set")
    
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    #logging.info(f"is cuda available on the device? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    for filename in os.listdir(in_files):
        img_name = filename.split("/")[-1]
        logging.info(f"Predicting image {filename} ...")
        #img_arr = np.load(filename)
        #img = Image.fromarray(img_arr)
        #img = Image.open(filename)
        img_path = os.path.join(in_files, filename)
        img = cv2.imread(img_path)
        start_time = time.time()
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        end_time = time.time()
        inference_time = end_time-start_time
        # pdb.set_trace()
        logging.info(f"Time to predict mask for image {img_name} is {inference_time}")

        if args.save:
            #print(f"out_files {out_files}")
            out_filename = os.path.join(args.output, filename)
            result = mask_to_image(mask, mask_values)
            #print('type result:', type(result))
            open_cv_image = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            print("type of open_cv_image: ", type(open_cv_image))
            print(str(os.path.join(args.output, filename)))
            cv2.imwrite(os.path.join(args.output, filename), cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
            
            # result = result.convert('RGB')
            # result.save(out_filename)
            logging.info(f"Mask saved to {out_filename}")

        if args.viz:
            logging.info(f"Visualizing results for image {filename}, close to continue...")
            plot_img_and_mask(img, mask)
