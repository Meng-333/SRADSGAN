import torch
import os, argparse

from model.srgan import SRGAN           # 自然图像SR模型2016
from model.edsr import EDSR             # 自然图像SR模型2017
from model.drcan import DRCAN           # 遥感影像SR模型2019
from model.dssr import DSSR             # 遥感影像SR模型2020
from model.sragan import SRAGAN         # 遥感影像SR模型2021
from model.ndsrgan import NDSRGAN       # 遥感影像SR模型2022
from model.amssrn import AMSSRN         # 遥感影像SR模型2022
from model.hat import HAT               # 自然图像SR模型2023
from model.sradsgan import SRADSGAN     # 本文提出的方法
from utils.utils import mkdir_and_rename

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of SR collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='NDSRGAN',
                        choices=['EDSR', 'SRGAN', 'DRCAN', 'DSSR', 'SRAGAN', 'NDSRGAN', 'AMSSRN', 'HAT', 'SRADSGAN'],
                        help='The type of model')
    parser.add_argument('--root_dir', type=str, default='/home/zju/mfe/SRADSGAN/')
    parser.add_argument('--data_dir', type=str, default='/home/zju/mfe/dataset/sradsgan/')
    parser.add_argument('--train_dataset', type=list, default=["AID", "DOTA", "LoveDA", "RSSCN7_2800", "SECOND"], choices=["AID", "DOTA", "LoveDA", "RSSCN7_2800", "SECOND"],
                        help='The name of training dataset')
    parser.add_argument('--test_dataset', type=list,
                        default=["UCMerced_LandUse"],
                        help='The name of test dataset')
    parser.add_argument('--crop_size', type=int, default=216, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=3, help='The number of channels to super-resolve')
    parser.add_argument('--scale_factor', type=int, default=4, help='Size of scale factor')
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--num_epochs', type=int, default=47, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=1, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--save_dir', type=str, default='Result', help='Directory name to save the results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate default 0.0002')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.99, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--gpu_mode', type=bool, default=True)

    parser.add_argument('--test_crop_size', type=int, default=216, help='Size of cropped HR image')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--hr_height', type=int, default=216, help='size of high res. image height')
    parser.add_argument('--hr_width', type=int, default=216, help='size of high res. image width')
    parser.add_argument('--sample_interval', type=int, default=1000,
                        help='interval between sampling of images from generators')
    # wgan & wgan_gp
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Loss weight for gradient penalty')
    parser.add_argument('--gp', type=bool, default=True, help='gradient penalty')
    parser.add_argument('--penalty_type', type=str, default='LS', choices=["LS", "hinge"], help='gradient type')
    parser.add_argument('--grad_penalty_Lp_norm', type=str, default='L2', choices=["L2", "L1", "Linf"], help='gradient penalty Lp norm')
    parser.add_argument('--relativeGan', type=bool, default=False, help='relative GAN')
    parser.add_argument('--loss_Lp_norm', type=str, default='L1', choices=["L2", "L1"], help='loss Lp norm')
    parser.add_argument('--weight_content', type=float, default=6e-3, help='Loss weight for content loss　')
    parser.add_argument('--weight_gan', type=float, default=1e-3, help='Loss weight for gan loss')

    parser.add_argument('--max_train_samples', type=int, default=40000, help='Max training samples')
    parser.add_argument('--is_train', type=bool, default=True, help='if at training stage')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if args.epoch == 0:
        mkdir_and_rename(os.path.join(args.root_dir, args.save_dir))  # rename old experiments if exists
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    if args.model_name == 'EDSR':
        net = EDSR(args)
    elif args.model_name == 'SRGAN':
        net = SRGAN(args)
    elif args.model_name == 'DRCAN':
        net = DRCAN(args)
    elif args.model_name == 'DSSR':
        net = DSSR(args)
    elif args.model_name == 'SRAGAN':
        net = SRAGAN(args)
    elif args.model_name == 'NDSRGAN':
        net = NDSRGAN(args)
    elif args.model_name == 'AMSSRN':
        net = AMSSRN(args)
    elif args.model_name == 'HAT':
        net = HAT(args)
    elif args.model_name == 'SRADSGAN':
        net = SRADSGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)

    # train
    net.train()

    # test
    # 测试集整体测试
    # net.mfeNew_validate(epoch=47, modelpath='./Result/NDSRGAN/model/generator_param.pkl')

    # 测试集下不同类别逐个测试
    net.mfeNew_validateByClass(47, save_img=True, modelpath="./Result/NDSRGAN/model/generator_param.pkl")

    # # 单张图片推理
    # #GF2_LR.tif scale_factor=3 crop_size=test_crop_size=hr_height=hr_width=72
    # net.mfe_test_single(img_fn="./img/GF2_LR.tif", modelpath="/home/zju/mfe/SRADSGAN/Result/NDSRGAN_x3/model/generator_param.pkl")

    # #Sentinel2.tif scale_factor=9 crop_size=test_crop_size=hr_height=hr_width=216
    # net.mfe_test_single(img_fn="./img/Sentinel2.tif", modelpath="/home/zju/mfe/SRADSGAN/Result/NDSRGAN_x9/model/generator_param.pkl")

if __name__ == '__main__':
    main()