"""
Inference script for fusion of RGB and NIR images. 
This script is based on DenseFuse inference script.

"""
# Standard imports
import torch
import numpy as np
import os
import sys
import argparse
import yaml
import torchvision.transforms as transforms
from torchvision.utils import save_image
from DataLoaderForest import InferenceDataset
from torch.utils.data import DataLoader

# Custom imports
from RES_VAE import VAE

# Argparser 
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default="inference.yml",
                    help="Path to the folder containing inference config file", required=False)

args = parser.parse_args()

def get_fused_image(model, rgb_image, nir_image, fusion_type="addition"):

    rgb_image = rgb_image.cuda()
    nir_image = nir_image.cuda()

    encode_rgb, _, _ = model.encoder(rgb_image)
    encode_nir, _, _ = model.encoder(nir_image)

    if fusion_type == "addition":
        encoded = encode_rgb+encode_nir
        
    fused_image = model.decoder(encoded)
    return fused_image

if __name__ == "__main__":

    # Try and read data from inference config yaml file.
    with open(args.config_file, "r") as f:
        try: 
            file = yaml.safe_load(f) 
        except yaml.YAMLError as load_error:
            print(load_error)

    # Check if the output folder exists. If not, create one.
    output_dir = os.path.join(file["output folder"], 
                             file["experiment name"])
    os.makedirs(output_dir, exist_ok=True
                             )

    # Transforms for dataloader
    transform = transforms.Compose([transforms.Resize(file["img size"]),
                                    transforms.ToTensor(),
                                    transforms.CenterCrop(file["img size"]),
                                    transforms.Normalize(0.5, 0.5)
                                ])
    
    inference_data = InferenceDataset(folder_nir_path=file["NIR img folder"], 
                                   folder_rgb_path= file["RGB img folder"],
                                   read_mode= file["img mode"], 
                                   transforms=transform)  
    
    inference_loader = DataLoader(inference_data, batch_size=1, shuffle=True)
    
    # The below two commands will return an iterable dictionary from the dataloader object above.
    # This is required because it is difficult to access the DataLoader object directly.
    detailer =iter(inference_loader)
    inference_image = next(detailer)

    # Load trained model
    vae_net = VAE(channel_in = inference_image["rgb_img"][0].shape[0],
                  ch = 128,
                  latent_channels=256)
    vae_net.load_state_dict(torch.load(file["model path"])["model_state_dict"])
    vae_net.eval()
    vae_net.cuda()

    for idx, images in enumerate(inference_loader):

        # Get RGB and NIR images from data-loader
        rgb_image = images["rgb_img"]
        nir_image = images["nir_img"]

        img_name = f"{idx}.png"

        fused_image = get_fused_image(vae_net, rgb_image, nir_image)
        save_image(fused_image, os.path.join(output_dir, str(img_name)))
        print("Saved image: ", img_name)

        


    
        