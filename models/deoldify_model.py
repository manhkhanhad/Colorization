import torch
from .base_model import BaseModel
from . import deoldify_net 
from . import networks 
from skimage import color  # used for lab2rgb
import numpy as np
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

class deoldifyModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """     
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(dataset_mode='aligned')
        #parser.set_defaults(netG='resnet_9blocks')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        body = create_body(resnet18, pretrained=True, n_in=1, cut=-2)
        self.netG = DynamicUnet(body, 3, (256, 256),self_attention=True).to(self.device)
        #self.netG.load_state_dict(torch.load("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/DeOldify_base_pix2pix/checkpoints/DeOldify_Landmark_RGB_noPretrainResnet/latest_net_G_pretrain.pth"))
        # print(self.netG)
        #backbone = deoldify_net.defineBackbone("resnet50",pretrained=True)
        # self.netG = deoldify_net.DynamicUnet(backbone,num_in_channels = 1, num_output_channels=2, input_size=(256,256))
        # self.netG = networks.init_net(self.netG,gpu_ids=self.gpu_ids)
        #self.netG = networks.define_G(1, 3, opt.ngf, 'unet_256', opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG.load_state_dict(torch.load("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/DeOldify_base_pix2pix/checkpoints/Pix2Pix_Landmark_RGB_PretrainGen/latest_net_G_pretrain.pth"))
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(4, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_pretrain = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100#* self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self,epoch):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def pretrain(self):
        self.forward()
        self.optimizer_G_pretrain.zero_grad()        # set G's gradients to zero
        self.backward_G_L1()                   # calculate graidents for G
        self.optimizer_G_pretrain.step()

    def backward_G_L1(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_GAN = 0
        self.loss_D_fake = 0
        self.loss_D_real = 0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()


    # def optimize_parameters(self,epoch):
    #     if epoch <= 100:
    #         self.forward()
    #         self.optimizer_G.zero_grad()        # set G's gradients to zero
    #         self.backward_G_L1()                   # calculate graidents for G
    #         self.optimizer_G.step()
    #     else:
    #         self.forward()                   # compute fake images: G(A)
    #         # update D
    #         self.set_requires_grad(self.netD, True)  # enable backprop for D
    #         self.optimizer_D.zero_grad()     # set D's gradients to zero
    #         self.backward_D()                # calculate gradients for D
    #         self.optimizer_D.step()          # update D's weights
    #         # update G
    #         self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
    #         self.optimizer_G.zero_grad()        # set G's gradients to zero
    #         self.backward_G()                   # calculate graidents for G
    #         self.optimizer_G.step()             # udpate G's weights
    
    def lab2rgb(self, L, AB):
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

    # def compute_visuals(self):
    #     """Calculate additional output images for visdom and HTML visualization"""
    #     self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)
    #     self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)

    #def load_networks(self,epoch):
        #self.netG.load_state_dict(torch.load("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/DeOldify_base_pix2pix/checkpoints/Pix2Pix_Landmark_RGB_PretrainGen/latest_net_G.pth"))