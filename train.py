import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.RL import RFAFN_0,RFAFN_234,RFAFN_134
from model.CFEB import RFAFN_DRB,RFAFN_CB
from model.GHFFS import RFA
from model.AFFB import SK,ESA
from model.model1 import IMDN,VDSR,lapsrn,CARN,RFDN
from data import traindataset,testdataset
import utils
import skimage.color as sc
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="RFAFN")
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=600,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=100,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default='1', type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="dataset",#变化
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,#6400
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default='ablation/LAST/RFAFN134_x4/epoch_599.pth', type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

#使用cuda
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = traindataset.div2k(args)#bianhua
testset =testdataset.DatasetFromFolderVal("Test_Datasets/test1/",
                                       "Test_Datasets/test{}/".format(args.scale),
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True
model=RFAFN_134.LFIFN(scale=args.scale)

l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:#迁移至默认设备进行训练
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        model_dict = model.state_dict()
        checkpoint = torch.load(args.pretrained)
        temp={}
        for k, v in checkpoint.items():
            try:
               if np.shape(model_dict[k]) == np.shape(v):
                   temp[k] = v
            except:
               pass
        model_dict.update(temp)
        model.load_state_dict(model_dict)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")
#初始化优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss = []
epoch_t = []

#定义train函数
def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    loss_epoch = utils.AverageMeter()  # 统计损失函数
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()

        loss_epoch.update(loss_sr.item(), lr_tensor.size(0))
        loss.append(loss_epoch.avg)
        epoch_t.append(epoch)

        #a1 = open(r'results/psnr/RFAFNx4.txt', 'a')
        if iteration % 500 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),loss_l1.item()))
            #print("{:.4f}".format(loss_l1), file=a1)
            #a1.close()

def valid():
    model.eval()
    global bestpsnr_epoch, bestssim_epoch
    global best_psnr, best_ssim

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
        psnr=avg_psnr / len(testing_data_loader)
        ssim=avg_ssim / len(testing_data_loader)
        if psnr > best_psnr:
            bestpsnr_epoch = epoch
            best_psnr = psnr
        if ssim > best_ssim:
            bestssim_epoch = epoch
            best_ssim = ssim
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(psnr, ssim))
    #f1 =open(r'ablation/RL/RFAFN_x3.txt','a')
    #f2 =open(r'ablation/RL/RFAFN_x3.txt','a')
    print('===> best_psnr epoch: {:.4f}, bestpsnr: {:.4f}'.format(bestpsnr_epoch - 1, best_psnr))
    #f1.close()
    print('===> best_ssim epoch: {:.4f}, bestssim: {:.4f}'.format(bestssim_epoch - 1, best_ssim))
    #f2.close()

def save_checkpoint(epoch):
    model_folder = "ablation/AFFB/SK2/"
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
bestpsnr_epoch, bestssim_epoch = 0, 0
best_psnr, best_ssim = 0.0, 0.0

for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid()
    train(epoch)
    save_checkpoint(epoch)






