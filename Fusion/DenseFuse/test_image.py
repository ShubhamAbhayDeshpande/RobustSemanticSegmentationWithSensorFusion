# test phase
import torch
from torch.autograd import Variable
from net import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import random
import sys


# Import L1 norm fusion strategy
from l1_fustion_strategy import L1_norm


def load_model(path, input_nc, output_nc):
    nest_model = DenseFuse_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print(
        "Model {} : params: {:4f}M".format(
            nest_model._get_name(), para * type_size / 1000 / 1000
        )
    )

    nest_model.eval()
    nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    # test = torch.unsqueeze(img_ir[:, i, :, :], 1)
    en_r = model.encoder(img1)
    # vision_features(en_r, 'ir')
    en_v = model.encoder(img2)
    # vision_features(en_v, 'vi')
    # fusion
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    # f = en_v
    # decoder
    img_fusion = model.decoder(f)
    return img_fusion[0]


def run_demo(
    model,
    infrared_path,
    visible_path,
    output_path_root,
    index,
    network_type,
    strategy_type,
    ssim_weight_str,
    mode,
):
    # Get the device from the computer. If the device is cuda move the tensor to device later.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if mode == 'L'

    # 'mode' argument is for number of channels in the image (gray or RGB)
    ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
    vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
    # else:
    # 	img_ir = utils.tensor_load_rgbimage(infrared_path)
    # 	img_ir = img_ir.unsqueeze(0).float()
    # 	img_vi = utils.tensor_load_rgbimage(visible_path)
    # 	img_vi = img_vi.unsqueeze(0).float()

    # dim = img_ir.shape
    if args.cuda:
        ir_img = ir_img.cuda()
        vis_img = vis_img.cuda()
        
    ir_img = Variable(
        ir_img, requires_grad=False
    )  # create a tensor which cannot have gradients
    vis_img = Variable(
        vis_img, requires_grad=False
    )  # Create a tensor which cannot have gradients
    dimension = ir_img.size()

    img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
    ############################ multi outputs ##############################################
    file_name = (
        "fusion_"
        + str(strategy_type)
        + "_"
        + str(index)
        + "_network_"
        + network_type
        + "_"
        + strategy_type
        + "_"
        + ssim_weight_str
        + ".png"
    )
    output_path = os.path.join(output_path_root, file_name)
    # # save images
    # utils.save_image_test(img_fusion, output_path)
    # utils.tensor_save_rgbimage(img_fusion, output_path)
    if args.cuda:
        img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = img_fusion.clamp(0, 255).data[0].numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    utils.save_images(output_path, img)

    print(output_path)


def vision_features(feature_maps, img_type):
    count = 0
    for features in feature_maps:
        count += 1
        for index in range(features.size(1)):
            file_name = (
                "feature_maps_"
                + img_type
                + "_level_"
                + str(count)
                + "_channel_"
                + str(index)
                + ".png"
            )
            output_path = "outputs/feature_maps/" + file_name
            map = features[:, index, :, :].view(
                1, 1, features.size(2), features.size(3)
            )
            map = map * 255
            # save images
            utils.save_image_test(map, output_path)


def main():
    # Comment the below line out when GPU usage is not a problem.
    torch.cuda.set_device(0)
    # run demo
    # test_path = "images/test-RGB/"
    # test_path = "images/IV_images/"

    test_path = (
        args.test_folder_path
    )  # "/home/deshpand/noadsm/datasets/Uni-dataset/001"

    network_type = "densefuse"
    strategy_type_list = [
        "addition",
        "attention_weight",
    ]  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    # Store the output in the following location. If the folder already does not exists, create one. 
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    strategy_type = strategy_type_list[0]  # strategy_type_list[0]

    # if os.path.exists(output_path) is False:
    #     os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = args.channels
    if in_c == 1:
        out_c = in_c
        mode = "L"
        model_path = args.model_path_gray
    else:
        out_c = in_c
        mode = "RGB"
        model_path = args.model_path_rgb

    with torch.no_grad():
        print("SSIM weight ----- " + args.ssim_path[2])
        ssim_weight_str = args.ssim_path[2]
        model = load_model(model_path, in_c, out_c)

        # Load the images from given test sample paths
        rgb_img_path = os.path.join(test_path, "rgb")
        nir_img_path = os.path.join(test_path, "NIR")

        rgb_names_all = [
            f
            for f in os.listdir(rgb_img_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        rgb_imgs_all = [os.path.join(rgb_img_path, img) for img in rgb_names_all]

        # random_imgs = random.sample(rgb_imgs_all, k=10)
        counter = 0
        for img in rgb_imgs_all:
            img_name = img.split("/")[-1].split(".")[0]
            nir_img_name = img_name + ".jpg"
            nir_img = os.path.join(nir_img_path, nir_img_name)
            print(nir_img)
            run_demo(
                model=model,
                infrared_path=nir_img,
                visible_path=img,
                output_path_root=output_path,
                index=img_name,
                network_type=network_type,
                strategy_type=strategy_type,
                ssim_weight_str=ssim_weight_str,
                mode=mode,
            )
            counter+=1
            print(counter)

        # for i in range(1):
        # 	index = i + 1
        # 	infrared_path = os.path.join(test_path + 'IR' + str(index) + '.jpg')
        # 	visible_path = test_path + 'VIS' + str(index) + '.jpg'
        # 	run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
    print("Done......")


if __name__ == "__main__":
    main()
