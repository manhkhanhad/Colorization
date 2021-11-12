"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
import torch
import os
import numpy as np
from skimage import color
from util import util
from models import deoldify_net 
from models import networks

def lab2rgb(L, AB):
    """Convert an Lab tensor image to a RGB numpy output
    Parameters:
        L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

    Returns:
        rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    """
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, AB2], dim=1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

def save_networks(net,save_path,gpu_ids):
    """Save all the networks to the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_path)
        net.cuda(gpu_ids[0])
    else:
        torch.save(net.cpu().state_dict(), save_path)

def pretrainGenerator(num_epoch, save_freq, save_dir, opt):
    #Setup network
    backbone = deoldify_net.defineBackbone("resnet50",pretrained=True)
    netG = deoldify_net.DynamicUnet(backbone,num_in_channels = 1, num_output_channels=2, input_size=(256,256))
    netG = networks.init_net(netG,gpu_ids=opt.gpu_ids)
    
    #Setup optimizer
    optimizer = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    #Setup dataset  
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    
    print('The number of training images = %d' % dataset_size)

    #Setup device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    #Setup result dir
    img_dir = os.path.join(save_dir, 'images_pretrainG')
    os.makedirs(img_dir, exist_ok = True) 

    #Setup loss function
    #L1 loss
    criterionL1 = torch.nn.L1Loss()

    iter = 0
    for epoch in range(num_epoch+1):
        epoch_start_time = time.time()
        
        for i, data in enumerate(dataset):
            iter += opt.batch_size
            input_img = data['A'].to(device)  # Set input
            real_img =  data['B'].to(device)
            preds = netG(input_img)
            loss = criterionL1(preds, real_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % opt.print_freq == 0:
                print("Epoch {}/{}: L1_loss = {}".format(epoch,num_epoch,loss) )
            
            #save image
            if iter % opt.display_freq:
                real = lab2rgb(input_img,real_img)
                fake = lab2rgb(input_img,preds)
                
                for image,label in zip([real,fake],["real","fake"]): #zip image and their label
                    image_numpy = util.tensor2im(image)
                    img_path = os.path.join(img_dir, 'epoch_pretrainG%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)

        if epoch % save_freq == 0:
            save_networks(netG, "{}/pretrain_netG_{}.pt".format(save_dir,epoch),opt.gpu_ids)
        save_networks(netG, "{}/pretrain_netG_latest.pt".format(save_dir,epoch),opt.gpu_ids)




if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Stage1: Pretrain Generator
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    pretrainGenerator(100,100,save_dir,opt)




