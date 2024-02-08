# -*- coding: utf-8 -*-
from glob import glob
from densefuse_net import DenseFuseNet
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from channel_fusion import channel_f as channel_fusion
from utils import mkdir, Strategy
import logging
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="cuda",
    required=True,
    help="Device for running the mdel. Either cuda or cpu.",
)
parser.add_argument(
    "--weights",
    default="./train_result/H_best.pkl",
    required=True,
    help="Training weights for encoder and decoder",
)
parser.add_argument(
    "--fusion", default="add", required=True, help="Fusion strategy to be used."
)
parser.add_argument(
    "--ir_img", default="./Test_ir/", required=True, help="Where ir images are stored."
)
parser.add_argument(
    "--rgb_img",
    default="./Test_vi/FLIR/",
    required=True,
    help="Where the rgb images are stored.",
)
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="test_log.log",
        filemode="a",
        format="[DualBranch::%(asctime)s]::%(message)s",
    )
    logging.info(
        r"==========================================Starting Testing====================================================="
    )

    _tensor = transforms.ToTensor()
    _pil_gray = transforms.ToPILImage()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = args.device
    # device = 'cpu'
    logging.info(f"device: {args.device}")
    model = DenseFuseNet().to(
        args.device
    )  # For any fusion stratergy, we need to perform encoding step on ir and rgb images. That is why we need this.
    checkpoint = torch.load(args.weights)
    # checkpoint = torch.load('./train_result/model_weight_new.pkl')
    model.load_state_dict(checkpoint["weight"])

    mkdir("outputs/fea/")
    mkdir("outputs/fea/vi/")
    mkdir("outputs/fea/ir/")
    mkdir("result")
    # test_ir = './Test_ir/'
    # test_vi = './Test_vi/'

    test_ir = args.ir_img
    test_vi = args.rgb_img

    def load_img(img_path, img_type="gray"):
        img = Image.open(img_path)
        if img_type == "gray":
            img = img.convert("L")
        return _tensor(img).unsqueeze(0).to(device)

    def normalize_input(image):
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        std = image.std(dim=(1, 2, 3), keepdim=True)
        img_normalized = (image - mean) / std
        return img_normalized

    fusename = [
        "l1",
        "add",
        "channel",
    ]  # These are the types of the fusion stratergies that we can use. For more info read litreature.
    if args.fusion not in fusename:
        ValueError("Incorrect fusion method entered.")

    def test(model):
        img_list_ir = glob(os.path.join(test_ir, "*.*"))
        img_num = len(img_list_ir)
        print("Test images num", img_num)
        logging.info(f"number of images in test folder: {len(img_list_ir)}")
        for img in img_list_ir:
            img_with_ext = img.split("/")[-1]
            img_name = img_with_ext.split(".")[0]
            print(img_with_ext)
            img_name = img_with_ext.split(".")[0]
            # The following are for the original dataset.
            # img1_path = test_ir+str(i)+'.bmp'
            # img2_path = test_vi+str(i)+'.bmp'
            # This is for the genaral datasets such as freiburg forest and others.
            # img1_path = test_ir+str(i)+'.png'
            img1_path = os.path.join(test_ir, img_with_ext)
            # img2_path = test_vi+str(i)+'.png'
            vi_img = img_name + ".jpg"
            img2_path = os.path.join(test_vi, vi_img)
            (
                img1,
                img2,
            ) = load_img(
                img1_path
            ), load_img(img2_path)

            img_1_normalized = normalize_input(img1)
            img_2_normalized = normalize_input(img2)
            s_time = time.time()
            feature1, feature2 = model.encoder(
                img_1_normalized, isTest=True
            ), model.encoder(img_2_normalized, isTest=True)
            # for name in fusename:
            if args.fusion == "channel":
                features = channel_fusion(
                    feature1, feature2, is_test=True
                )  # perform channel fusion before giving the output to the decoder.
                out = (
                    model.decoder(features).squeeze(0).detach().cpu()
                )  # Do decoding on CPU? why? Can we use GPU here instead?
                save_name = (
                    "result/" + args.fusion + "/fusion_" + str(img_name) + ".jpg"
                )
                mkdir("result/" + args.fusion)
                img_fusion = _pil_gray(out)
                img_fusion.save(save_name)
                # print("pic:[%d] %.4fs %s"%(img_name,e_time,save_name))
                logging.info(f"Did fusion for img: {img_name}")
            else:
                """
                If the fusion strategy is not channel, we want to do it with simple addition strategy. This is what is shown below.

                """
                with torch.no_grad():
                    fusion_layer = Strategy(args.fusion, 1).to(device)
                    feature_fusion = fusion_layer(feature1, feature2)
                    out = model.decoder(feature_fusion).squeeze(0).detach().cpu()
                e_time = time.time() - s_time
                # save_name = 'result/'+name+'/fusion'+str(i)+'.bmp'
                save_name = (
                    "result/" + args.fusion + "/fusion_" + str(img_name) + ".jpg"
                )
                mkdir("result/" + args.fusion)
                img_fusion = _pil_gray(out)
                img_fusion.save(save_name)
                # print("pic:[%d] %.4fs %s"%(img_name,e_time,save_name))
                logging.info(f"Did fusion for img: {img_name}")

    with torch.no_grad():
        test(model)
