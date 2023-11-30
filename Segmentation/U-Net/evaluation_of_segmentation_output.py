"""

Script to predict the dice score with the help of actual and predicted segmentation map of the images. 

The score will be stored in a log file.

"""
#Imports
import argparse
import logging
import torch
import cv2 
import os
parser = argparse.ArgumentParser()

parser.add_argument("--original_maps", default="", required=False, help="location to the original maps")
parser.add_argument("--predicted_maps", default="", required=False, help="location of the predicted maps")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, filename="dice_score_prediction.log", filemode="a", format="%(message)s")

if __name__ == "__main__":

    pass