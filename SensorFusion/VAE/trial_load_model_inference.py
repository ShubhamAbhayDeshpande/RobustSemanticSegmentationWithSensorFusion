"""
This file is a trial to see how we can load and parse the model for doing inference on the test dataset for fusion. 

For inference, we can use the DenseFuse model inference file as a reference.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from RES_VAE import VAE

modle_path = "Models/Freiburg_Forest_128x218_epochs_50_128_copy.pt"

# Load the model
vae_net = VAE(channel_in=3, ch=128, latent_channels=256)
vae_net.load_state_dict(torch.load(modle_path)["model_state_dict"])
vae_net.eval()
vae_net.cuda()

print(vae_net.__dir__())

for idx, elements in enumerate(vae_net.__dir__()):
    if "decoder" in elements:
        print(idx, elements, sep=" : ")
