"""
This script will check the validation dice score for masks predicted for the forest images.

"""

import cv2
import os
import argparse
import logging
import torch
import torchvision.transforms as transforms

from utils.dice_score import dice_coeff

# Input parser
parser = argparse.ArgumentParser()
parser.add_argument('--GT',
                    type=str,
                    required=False,
                    default="/home/deshpand/noadsm/datasets/Uni-dataset/final_dataset/test/data/masks",
                    help="Path where the ground truth masks.",
                    )
parser.add_argument('--masks',
                    type=str,
                    required=False,
                    default="/home/deshpand/Thesis/semantic_segmentation_network/Pytorch-UNet/output/NDVI_Uni-dataset_ssim_wt_1000000_lr_1e-3/dice_score_check",
                    help="Path where the results of the model are saved",
                    )
args = parser.parse_args()

if __name__ == "__main__":

    # Create a logger to log messages while execution
    logging.basicConfig(filename="validation_prediction.log", 
                        filemode="a",
                        format="%(asctime)s: %(levelname)s: %(message)s",
                        level=logging.INFO
                        )      
    
    logging.info("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    img_dice_score = dict()
    logging.info(f"gt image path: {args.GT} \ninference image path: {args.masks}")
    for gt_img_name in os.listdir(args.GT):

        img_name = gt_img_name.split('.')[0]
        
        # Check if the file extension is '.png' and change the file extension for image name if needed.
        if os.path.splitext(gt_img_name)[-1]=='.png':
            path_img = os.path.join(args.masks, gt_img_name)
            if os.path.exists(path_img):
                pass
            else:
                img_name = gt_img_name.split('.')[0] + ".jpg"
                path_img = os.path.join(args.masks, img_name)

        # Get the GT image path and predicted masks paths
        gt_img_path = os.path.join(args.GT, gt_img_name)
        

        
        # Read images using openCV. 
        gt_img = cv2.imread(str(gt_img_path))
        img = cv2.imread(str(path_img))

        # Convert read images from BGR to RGB
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        if gt_img is None:
            raise ValueError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise ValueError

        # Convert images to tensors
        transform = transforms.Compose([transforms.ToTensor()])
        gt_img_tensor = transform(gt_img)
        img_tensor = transform(img)

        inter = 2 * (img_tensor*gt_img_tensor).sum()
        sets_sum = img_tensor.sum() + gt_img_tensor.sum()
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        # Dice score calculation
        dice = (inter + 1e-6)/(sets_sum + 1e-6)

        img_dice_score[gt_img_name] = dice.item()
    
    print(f"Dice score for images: \t{img_dice_score}")
    logging.info(f"Dice score for images: \n\t{img_dice_score}")
    