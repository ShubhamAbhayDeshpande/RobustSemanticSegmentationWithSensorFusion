"""
Program modified so that, it will normalize all the recreated images returned by the encoder-decoder architecture. To see if this helps with the loss calulation 
and overall output.

Following notes identify what is different in each iteration of the program.

V_0.1 :: Chanded the weight associated with perceptual loss to 1, learning rate changed to 1e-2, stored recreated images for visualization, 
        changed dataset for nir to nir_gray. 
V_0.2 :: Added weights to the individual losses in the equations as suggested in the literature. 
        (The values of the weights may change in the future iterations) Select the value such that all losses are represented equally in the sum.
"""
# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import torch
from tqdm import tqdm
from densefuse_net import DenseFuseNet
from ssim import SSIM
from utils import mkdir, AEDataset, gradient, hist_similar
import os
import time
from loss_network import LossNetwork
import logging
import argparse
import scipy.io as scio
import sys

parser = argparse.ArgumentParser()

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./normalized_input_our_data")

parser.add_argument(
    "--device",
    default="cpu",
    required=True,
    help="Device for running the mdel. Either 'cuda' or 'cpu'.",
)
parser.add_argument(
    "--ir_img", default="./data/ir", required=True, help="Where ir images are stored."
)
parser.add_argument(
    "--rgb_img",
    default="./data/rgb",
    required=True,
    help="Where the rgb images are stored.",
)
parser.add_argument("--epochs", default=100, required=True, help="Number of epochs.")
parser.add_argument("--batch_size", default=10, required=True, help="Batch size.")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    filename="normalized_input_our_data_train.log",
    filemode="w",
    format="[DualBranchNetworkTrain::%(asctime)s:]:%(message)s",
)
# import scipy.io as scio
# import numpy as np
# from matplotlib import pyplot as plt


def save_regenerated_img(torch_tensor, epoch_int, index_int):
    """
    This function will be used to save any random image from a given tensor of images. Note: we are using the index and the epoch only for name of the image.

    param torch_tensor: Multidimensional tensor containing the recreated images from the network.
    param epoch_int: Epoch in which the image was generated.
    param index_int: Index of the epoch in which the image was generated.

    """
    img_name = (
        "./normalized_our_data" + "/" + str(epoch_int) + "_" + str(index_int) + ".png"
    )
    print(f"Saved reconstructed image under location: {img_name}")
    rand_idx_tensor = torch.randint(0, 9, (1,))
    rand_idx = rand_idx_tensor.item()
    rand_img_tensor = torch_tensor[rand_idx, :, :, :]
    rand_img = rand_img_tensor.squeeze(0).detach().cpu()
    tensor_img_norm = (rand_img - rand_img.min()) / (rand_img.max() - rand_img.min())
    vutils.save_image(tensor_img_norm, img_name)


if __name__ == "__main__":
    logging.info(
        r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    logging.info("Starting training")
    save_name = "H_normalized_our_data"
    os.chdir(r"./")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Loss weights
    # These weights are defined empirically. The assumption was that all values of losses should be close to 0.5
    coeff_mse = 20
    coeff_grad = 10
    coeff_hist = 0.2
    coeff_perception = 16

    # Parameters

    root_ir = args.ir_img
    root_vi = args.rgb_img
    logging.info(f"Training images for rgb :{root_vi}")
    logging.info(f"Training images for nir :{root_ir}")
    train_path = "./train_result/"  # No need to change this. Will need this later on. Just try and add better file name with date inside this folder.
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    logging.info("epochs: ", epochs)
    logging.info("batch_size: ", batch_size)

    device = args.device
    lr = 1e-3  # Old learning rate.
    # lr = 1e-2
    logging.info(f"learning rate: {lr}")
    lambd = 1
    logging.info(f"lambda: {lambd}")

    # Dataset
    data_TNO = AEDataset(root_ir, root_vi, resize=256, gray=True)
    loader_TNO = DataLoader(
        data_TNO, batch_size=batch_size, shuffle=True
    )  # This will return the batches with batch size for the entire dataset.

    img_num = len(data_TNO)
    print("Load ---- {} pairs of images: TNO:[{}]".format(img_num, len(data_TNO)))

    # Model
    model = DenseFuseNet().to(device)
    # checkpoint = torch.load('./train_result/H_model_weight_new.pkl')
    # model.load_state_dict(checkpoint['weight'])
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=False,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-10,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    # Loss definition

    MSE_fun = nn.MSELoss()
    SSIM_fun = SSIM()  # SSIM code looks shady. Check this in case of error.
    CrossEntropyLoss = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()

    # Training

    # The losses in the literature while training are 1. Pixel 2. Gradient 3. Color 4. Perceptual.
    # Defining training loss
    mse_train = []
    ssim_train = []
    loss_train = []
    gradient_train = []

    # Defining Validation loss
    mse_val = []
    ssim_val = []
    loss_val = []

    mkdir(train_path)
    min_loss = 20  # This value will change based on the dataset and imge complexity.

    print("============ Training Begins [epochs:{}] ===============".format(epochs))
    # steps = len(loader_TNO)+len(loader_k)
    steps = len(loader_TNO)
    logging.info(f"length of dataloader: {steps}")
    s_time = time.time()  # start time of the training.
    loss = torch.zeros(
        1
    )  # Pre-defining loss as zero because we may need to updatae the sceduler for first time based on this loss.

    # The following losses are image specific and given in literature.
    grd_loss_all = []
    hist_loss_all = []
    mse_loss_all = []
    perceptual_loss_all = []

    for iteration in range(epochs):  # This will load the epochs.
        logging.info(f"{iteration}/{epochs}")
        scheduler.step(
            loss.item()
        )  # This will adjust the learining rate in case lr does not change after multiple iterations.
        img_name = "Freiburg_Forest"
        imgs_T = iter(
            loader_TNO
        )  # It will create an iterable for the number of batches given by the Dataloader class. Here it is represented by loader_TNO variable.
        _loss_per_epoch = []

        tqdms = tqdm(
            range(int(steps))
        )  # tqdm will provide the progress bar for each step.
        for index in tqdms:  # This will load the batches.
            img = next(imgs_T).to(device)  # This will send the image for validation.
            optimizer.zero_grad()  # Set all gradients of optimizer to zero.

            # giving the image normalized between 0 and 1 as input to the network
            mean = img.mean(dim=(1, 2, 3), keepdim=True)
            std = img.std(dim=(1, 2, 3), keepdim=True)
            img_normalized = (img - mean) / std

            img_re = model(
                img_normalized
            )  # Send original image to the encoder and decoder model and get reconstructed image back.
            # print(img_re.size())
            # print(type(img_re.size(dim=0)))
            # mean = img_re.mean(dim=(1,2,3), keepdim=True)
            # std = img_re.std(dim=(1,2,3), keepdim= True)
            # img_re_normalized = (img_re-mean)/std

            # img_re.to(device)
            # img.to(device)

            if iteration % 3 == 0 and index == 5:
                save_regenerated_img(img_re, iteration, index)

            # This is where we calculate all the losses by taking into account the recreated img and the original img.
            mse_loss = MSE_fun(img, img_re)
            logging.info(f"mse loss: {mse_loss}")
            # logging.info(f'weighted mse loss: {mse_loss*20}')
            grd_loss = MSE_fun(gradient(img), gradient(img_re))
            logging.info(f"grad loss: {grd_loss}")
            # logging.info(f'weighted grad loss: {grd_loss*10}')
            hist_loss = hist_similar(img, img_re.detach()) * 0.001
            logging.info(f"Standard loss (Histogram loss): {hist_loss}")
            # logging.info(f'weighted standard loss: {hist_loss*0.2}')
            # std_loss = torch.abs(img_re.std() - img.std())
            std_loss = hist_loss

            # Perceptual Loss Calculations.
            with torch.no_grad():
                x = img.detach()
            features = loss_network(
                x
            )  # Loss network refers to VGG net. Features extracted from this network (for real and reconstructed image) are used for Perceptual loss
            features_re = loss_network(img_re)

            with torch.no_grad():  # Check the number of channels in the output of the VGG network as it is not specifically mentioned in literature.
                f_x_vi1 = features[1].detach()
                f_x_vi2 = features[2].detach()
                f_x_ir3 = features[3].detach()
                f_x_ir4 = features[4].detach()

            perceptual_loss = (
                MSE_fun(features_re[1], f_x_vi1)
                + MSE_fun(features_re[2], f_x_vi2)
                + MSE_fun(features_re[3], f_x_ir3)
                + MSE_fun(features_re[4], f_x_ir4)
            )
            std_loss = std_loss
            # perceptual_loss = perceptual_loss
            perceptual_loss = (
                perceptual_loss * 1000
            )  # Initially multiplied by 1000. No idea what was the need to do that.
            logging.info(f"Perceptual loss: {perceptual_loss}")
            # logging.info(f'weighted perceptual loss: {perceptual_loss*16}')

            grd_loss_all.append(grd_loss.item())
            hist_loss_all.append(hist_loss.item())
            mse_loss_all.append(mse_loss.item())
            perceptual_loss_all.append(perceptual_loss.item())

            # loss = (mse_loss*20) +(grd_loss*10) + (std_loss*0.2)+ (perceptual_loss*16)
            loss = mse_loss + grd_loss + std_loss + perceptual_loss
            logging.info(f"Total Loss: {loss}")
            _loss_per_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

            e_time = time.time() - s_time
            last_time = (
                epochs * int(steps) * (e_time / (iteration * int(steps) + index + 1))
                - e_time
            )

            tqdms.set_description(
                "%d MSGP[%.5f %.5f %.5f %.5f] T[%d:%d:%d] lr:%.4f "
                % (
                    iteration,
                    mse_loss.item(),
                    std_loss.item(),
                    grd_loss.item(),
                    perceptual_loss.item(),
                    last_time / 3600,
                    last_time / 60 % 60,
                    last_time % 60,
                    optimizer.param_groups[0]["lr"],
                )
            )
            # scheduler.step(loss.item())

            # print('[%d,%d] -   Train    - MSE: %.10f, SSIM: %.10f '%
            #    (iteration,index,mse_loss.item(),ssim_loss.item()))
            # if iteration>1:
            #      mse_train.append(mse_loss.item())
            #      ssim_train.append(ssim_loss.item())
            #      loss_train.append(loss.item())
            #      gradient_train.append(grd_loss.item())

            # with torch.no_grad():
            # tmp1, tmp2 = .0, .0
            # for _, img in enumerate(loader_val):
            # img = img.to(device)
            # img_recon = model(img)
            # tmp1 += (MSE_fun(img,img_recon)*img.shape[0]).item()
            # tmp2 += (SSIM_fun(img,img_recon)*img.shape[0]).item()
            # tmp3 = tmp1+lambd*tmp2
            # mse_val.append(tmp1/data_val.__len__())
            # ssim_val.append(tmp1/data_val.__len__())
            # loss_val.append(tmp1/data_val.__len__())
            # print('[%d,%d] - Validation - MSE: %.10f, SSIM: %.10f'%
            # (iteration,index,mse_val[-1],ssim_val[-1]))
            # scio.savemat(os.path.join(train_path, 'TrainData.mat'),
            # {'mse_train': np.array(mse_train),
            #'ssim_train': np.array(ssim_train),
            #'loss_train': np.array(loss_train)})
            # scio.savemat(os.path.join(train_path, 'ValData.mat'),
            # {'mse_val': np.array(mse_val),
            #'ssim_val': np.array(ssim_val),
            #'loss_val': np.array(loss_val)})

        # plt.figure(figsize=[12,8])
        # plt.subplot(2,2,1), plt.semilogy(mse_train), plt.title('mse train')
        # plt.subplot(2,2,2), plt.semilogy(ssim_train), plt.title('ssim train')
        # plt.subplot(2,2,3), plt.semilogy(gradient_train), plt.title('grd val')
        # plt.subplot(2,2,4), plt.semilogy(loss_train), plt.title('loss train')
        # plt.subplot(2,2,4), plt.semilogy(mse_val), plt.title('mse val')
        # plt.subplot(2,3,5), plt.semilogy(ssim_val), plt.title('ssim val')
        # plt.subplot(2,3,6), plt.semilogy(loss_val), plt.title('loss val')

        # plt.savefig(os.path.join(train_path,'curve.png'),dpi=90)

        # The following are used to save the model weights when the current loss is less than the loss specified by min_loss.
        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(
                {
                    "weight": model.state_dict(),
                    "epoch": iteration,
                    "batch_index": index,
                },
                os.path.join(train_path, save_name + "best.pkl"),
            )
            print("[%d] - Best model is saved -" % (iteration))

        # This will save the model weights if the iteration divided by 10 is equal to 0.
        if (iteration + 1) % 10 == 0 and iteration != 0:
            torch.save(
                {
                    "weight": model.state_dict(),
                    "epoch": iteration,
                    "batch_index": index,
                },
                os.path.join(train_path, save_name + "model_weight_new.pkl"),
            )
            print("[%d] - model is saved -" % (iteration))

        # Adding the average loss after every 5 epochs to the tensor board along with the learining rate for last epoch.
        if iteration % 4 == 0:
            loss_board = sum(_loss_per_epoch) / steps
            writer.add_scalar("Train Loss", loss_board, iteration)
            writer.add_scalar(
                "Learning Rate", optimizer.param_groups[0]["lr"], iteration
            )

    scio.savemat(
        "./train_result/loss_all_normalized_our_data.mat",
        {
            "grd_loss_all": grd_loss_all,
            "hist_loss_all": hist_loss_all,
            "mse_loss_all": mse_loss_all,
            "perceptual_loss_all": perceptual_loss_all,
        },
    )
