import argparse
import os
import numpy as np
import utils
import skimage.color as sc
import cv2

from model.RL import RFAFN_134,RFAFN_24
from model.GHFFS import CC,Dense,RFA
from model.model1 import  EDSR_baseline,RFAFN_134,SRCNN
from model.AFFB import ESA,SK
from model.model1 import CARN,RFDN,IDN,IMDN
import torch.nn as nn
import torch
from thop import profile
from thop import clever_format
# Testing settings
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='IMDN')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/CTtest2',#IMAGE/test10/
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/CTtest2X2/',#IMAGE/test10_x4/
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/visual/x4/RFDN')
parser.add_argument("--checkpoint", type=str, default='sota/SRCNN/SRCNN2_x2/epoch_598.pth',#CT_x2/epoch_37.pth
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model=SRCNN.SRCNN()
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

def count_pixelshuffle(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements
    m.total_ops = torch.Tensor([int(total_ops)])
input = torch.randn(1, 3, 160, 160)
macs, params = profile(model, inputs=(input, ),custom_ops={nn.PixelShuffle:count_pixelshuffle})
macs, params = clever_format([macs, params], "%.2f")
print(params)
print(macs)


i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + ext,cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        #out=F.interpolate(im_input, scale_factor=opt.upscale_factor, mode='bicubic',align_corners=False)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    #output_folder = os.path.join(opt.output_folder,
                                 #imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    #cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1


print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
