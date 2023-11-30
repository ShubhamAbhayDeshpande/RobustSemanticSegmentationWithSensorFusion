# Standard Imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import argparse
from torchvision import models, transforms
import os

# Custom imports 
from unet import UNet, unet_parts
from utils.data_loading import BasicDataset


# Loading device. cuda or cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with weights and get the state dict
model = UNet(n_channels=3, n_classes=6, bilinear=False)
model.to(device=device, dtype=torch.float32)
state_dict = torch.load("checkpoints/full_uni_dataset_train_30_epochs_2_batch_size_1e-5_lr/best_model/best_model.pth", map_location=device)
mask_values = state_dict.pop("mask_values", [0, 1])
model.load_state_dict(state_dict)
model.eval()


# Need to go through the model to get the weights of the layers.
model_children = list(model.children())

# Use the following as a reference when parsing the model.
# print(type(model_children[0].double_conv[0])) # This wil iterate DoubleConv --> double_conv --> Conv2d.
# print(type(model_children[1].maxpool_conv[1].double_conv[0])) # Down --> MaxPool2d --> DoubleConv --> Conv2d.
# print(type(model_children[5].conv.double_conv[0]))  # Up --> conv --> double_conv --> Conv2d.
# print(type(model_children[-1].conv)) # OutConv --> Conv2d

# Counter to keep count of all convolution layers.
counter = 0

# Make empty list to store layers weight and layers name.
layer_weight = []
layer_name = [] 
layers = []

for i in range(len(model_children)):

    if type(model_children[i]) == unet_parts.DoubleConv:
        for j in range(len(model_children[i].double_conv)):
            if type(model_children[i].double_conv[j]) == nn.Conv2d:
                layer_weight.append(model_children[i].double_conv[j].weight)
                layer_name.append("DoubleConv_" + str(i) + "_" + str(j))
                layers.append(model_children[i].double_conv[j])

    elif type(model_children[i]) == unet_parts.Down:
        for j in range(len(model_children[i].maxpool_conv[1].double_conv)):
            if type(model_children[i].maxpool_conv[1].double_conv[j]) == nn.Conv2d:
                layer_weight.append(model_children[i].maxpool_conv[1].double_conv[j].weight)
                layer_name.append("Down_" + str(i) + "_" + str(j))
                layers.append(model_children[i].maxpool_conv[1].double_conv[j])

    elif type(model_children[i]) == unet_parts.Up:
        for j in range(len(model_children[i].conv.double_conv)):
            if type(model_children[i].conv.double_conv[j]) == nn.Conv2d:
                layer_weight.append(model_children[i].conv.double_conv[j].weight)
                layer_name.append("Up_" + str(i) + "_" + str(j))
                layers.append(model_children[i].conv.double_conv[j])
                
    else:
        layer_weight.append(model_children[i].conv.weight)
        layer_name.append("OutConv")
        layers.append(model_children[i].conv)


# read image and apply transform as we do in iference
img_path = "/home/deshpand/noadsm/datasets/Uni-dataset/final_dataset/test/data/imgs/0282.jpg"

full_img = cv2.imread(img_path)
img = torch.from_numpy(BasicDataset.preprocess(None, full_img, 0.5, is_mask=False))
img = img.unsqueeze(0)
img = img.to(device = device, dtype = torch.float32)

results = [layers[0](img)]
for i in range(1, len(layers)):
    new_result = layers[i](results[-1])
    new_results = (new_result-new_result.min())/ (new_result.max() - new_result.min()) * 255
    new_results_after_relu = torch.nn.functional.relu(new_results)
    results.append(new_results_after_relu)

# Make a copy of results 
outputs = results

# Visualize first 64 feature maps from the upper layer
for num_layer in range(len(outputs)):
    plt.figure(figsize=(60, 60))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    layer_viz = layer_viz.cpu()
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='hot')
        plt.axis("off")
        plt.savefig(f"./layer_{num_layer}.png")
        # plt.show()
        plt.close()
