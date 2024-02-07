# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils

# from utils import *
from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim
import wandb


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    original_imgs_path_rgb = utils.list_images(
        args.dataset_rgb
    )  # get image from file location
    print("Numner of rgb images: ", len(original_imgs_path_rgb))

    original_imgs_path_ir = utils.list_images(
        args.dataset_ir
    )  # Get nir image from file location
    print("Number of NIR images: ", len(original_imgs_path_ir))
    # Combining paths for rgb and ir images.
    original_imgs_path_combined = original_imgs_path_rgb + original_imgs_path_ir
    original_imgs_path = random.sample(
        original_imgs_path_combined, len(original_imgs_path_combined)
    )

    i = 5#0#5#0
    train(i, original_imgs_path)


def train(i, original_imgs_path):
    # Use wandb for logging the data related to run.
    experiment = wandb.init(
        # Setting project name where the run will be logged
        project="Multispectral_Fusion_DenseFuse",
        # Tracking the parameters
        config={
            "learning_rate": args.lr,
            "dataset": args.dataset[1],
            "epochs": args.epochs,
            "exp_name": args.exp_name,
        },
        # Adding the name of experiment to keep better track of the experiment.
        name=args.exp_name,
    )

    batch_size = args.batch_size

    # load network model, RGB
    in_c = args.channels  # 1 - gray; 3 - RGB
    if in_c == 1:
        img_model = "L"
    else:
        img_model = "RGB"
    input_nc = in_c
    output_nc = in_c
    densefuse_model = DenseFuse_net(input_nc, output_nc)  # Model definition.

    if args.resume is not None:
        print("Resuming, initializing using weight from {}.".format(args.resume))
        densefuse_model.load_state_dict(torch.load(args.resume))
    print(densefuse_model)
    optimizer = Adam(params=densefuse_model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        densefuse_model.cuda()

    tbar = trange(args.epochs)  # method for returning tqdm range.
    print("Start training.....")

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir, args.exp_name)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    total_loss_per_epoch = []
    all_ssim_loss = 0.0
    all_pixel_loss = 0.0
    all_gradient_loss = 0.0
    all_total_loss = 0.0
    for e in tbar:
        print("Epoch %d....." % e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        densefuse_model.train()
        count = 0
        ssim_loss_epoch_lost_list = []
        pixel_loss_epoch_list = []
        gradient_loss_epoch_list = []
        total_loss_epoch_list = []
        gradient_loss_list = []
        for batch in range(batches):
            image_paths = image_set_ir[
                batch * batch_size : (batch * batch_size + batch_size)
            ]
            img = utils.get_train_images_auto(
                image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model
            )

            count += 1
            optimizer.zero_grad()
            img = Variable(img, requires_grad=False)

            if args.cuda:
                img = img.cuda()
            # get fusion image
            # encoder
            en = densefuse_model.encoder(img)
            # decoder
            outputs = densefuse_model.decoder(en)
            # resolution loss
            x = Variable(img.data.clone(), requires_grad=False)
            ssim_loss_value = 0.0
            pixel_loss_value = 0.0
            gradient_loss_value = 0.0
            for output in outputs:
                # This loop will calculate the loss for individual image in batch.
                # print('output size: ', output.size())
                # print('x shape: ',x.size())
                pixel_loss_temp = mse_loss(output, x)
                ssim_loss_temp = ssim_loss(output, x, normalize=True)

                # adding gradient loss
                # Depending upon the number of channels in the image, the gradient loss function will change.
                if args.channels == 1:
                    grad_loss_temp = mse_loss(
                        utils.gradient_gray(output), utils.gradient_gray(x)
                    )
                if args.channels == 3:
                    grad_loss_temp = 0  # Cannot calculate gradient for a RGB image.

                ssim_loss_value += 1 - ssim_loss_temp
                pixel_loss_value += pixel_loss_temp
                # gradient_loss_value +=grad_loss_temp

            ssim_loss_value /= len(outputs)  # average ssim loss per batch.
            pixel_loss_value /= len(outputs)  # average pixel loss per batch.
            # gradient_loss_value /= len(output)  # average gradient loss per batch.

            # total loss
            total_loss = pixel_loss_value + (
                args.ssim_weight[i] * ssim_loss_value
            )  # + (gradient_loss_value*args.gradient_weight)
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()
            # try:
            # 	all_gradient_loss += gradient_loss_value.item()
            # except AttributeError:
            # 	all_gradient_loss = torch.tensor(gradient_loss_value).item()
            all_total_loss += total_loss.item()

            ssim_loss_epoch_lost_list.append(ssim_loss_value.item())
            pixel_loss_epoch_list.append(pixel_loss_value.item())
            # gradient_loss_epoch_list.append(gradient_loss_value.item())
            total_loss_epoch_list.append(total_loss.item())

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    batches,
                    all_pixel_loss / args.log_interval,
                    all_ssim_loss / args.log_interval,
                    (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss)
                    / args.log_interval,
                )
                tbar.set_description(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_all.append(
                    (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss)
                    / args.log_interval
                )

                all_ssim_loss = 0.0
                all_pixel_loss = 0.0

        mean_ssim_loss = sum(ssim_loss_epoch_lost_list) / len(ssim_loss_epoch_lost_list)
        mean_pix_loss = sum(pixel_loss_epoch_list) / len(pixel_loss_epoch_list)
        # mean_gradient_loss = sum(gradient_loss_epoch_list)/len(gradient_loss_epoch_list)
        mean_total_loss = sum(total_loss_epoch_list) / len(total_loss_epoch_list)

        wandb.log(
            {
                "Epoch": e,
                "SSIM Loss": mean_ssim_loss,
                "Pixel_Loss": mean_pix_loss,  # "Gradient loss": mean_gradient_loss,
                "Total loss": mean_total_loss,
                "Learning Rate": args.lr,
            }
        )

        # save best model based on average total loss.
        total_loss_per_epoch.append(mean_total_loss)
        if e == 0:
            pass
        else:
            if total_loss_per_epoch[e - 1] > total_loss_per_epoch[e]:
                # Create a folder to save model
                folder_for_model = os.path.join("models", "best_models")

                # if the folder does not exist, create it
                if not os.path.exists(folder_for_model):
                    os.mkdir(folder_for_model)

                model_name = args.exp_name + "_" + str(e) + ".model"
                model_path = os.path.join(folder_for_model, model_name)

                densefuse_model.eval()
                densefuse_model.cpu()

                torch.save(densefuse_model.state_dict(), model_path)

                if args.cuda:
                    densefuse_model.cuda()

                print("\nBest model saved at", model_path)


if __name__ == "__main__":
    main()
