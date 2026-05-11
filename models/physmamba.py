import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from ssim import SSIM
import time
from .unetgan.mamba_discriminator import MambaThermalDiscriminator as ThermalMambaDiscriminator
from .unetgan.self_perceptual_loss import SelfPerceptualLoss

class PhysMamba(BaseModel):
    def name(self):
        return 'PhysMamba'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu'
        )


        # 加载/定义网络
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid,
                opt.init_type, self.gpu_ids,
                resolution=256 if opt.dataset_mode == 'FLIR' else 512
            )

            self.netD_thermal_mamba = ThermalMambaDiscriminator(
                input_nc=opt.input_nc + opt.output_nc,
                embed_dim=opt.ndf * 2,
                depth=4,
                patch_size = 4
            ).to(self.device)

            if len(self.gpu_ids) > 0:
                self.netD_thermal_mamba = torch.nn.DataParallel(
                    self.netD_thermal_mamba, self.gpu_ids
                )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD_thermal_mamba, 'D_mamba', opt.which_epoch)

        self.fake_AB_pool = ImagePool(opt.pool_size)
        self.criterionGAN = networks.GANLoss(
            use_lsgan=not opt.no_lsgan, tensor=self.Tensor
        )
        self.criterionL1 = torch.nn.L1Loss()
        self.ssim = SSIM()


        if self.isTrain:
            self.schedulers = []
            self.optimizers = []

            self.criterionSelfPer = SelfPerceptualLoss(
                pretrained_path=opt.pretrained_encoder_path,
                device=self.device)

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr / 100, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D_thermal_mamba = torch.optim.Adam(
                self.netD_thermal_mamba.parameters(),
                lr=opt.lr / 100, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [
                self.optimizer_G,
                self.optimizer_D,
                self.optimizer_D_thermal_mamba
            ]
            self.schedulers = [
                networks.get_scheduler(opti, opt) for opti in self.optimizers
            ]

        print('---------- Networks initialized -------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            t = time.time()
            self.fake_B = self.netG(self.real_A)
            t = time.time() - t
            self.real_B = Variable(self.input_B)
        return t
    # get image paths
    def get_image_paths(self):
        return self.image_paths
    def backward_D(self):

        fake_AB = torch.cat((self.real_A, self.fake_B), 1).detach()
        real_AB = torch.cat((self.real_A, self.real_B), 1)


        pred_fake_local, pred_fake_bottleneck = self.netD(fake_AB)
        pred_real_local, pred_real_bottleneck = self.netD(real_AB)

        loss_D_conv_fake = (
            self.criterionGAN(pred_fake_local, False) +
            self.criterionGAN(pred_fake_bottleneck, False)
        )
        loss_D_conv_real = (
            self.criterionGAN(pred_real_local, True) +
            self.criterionGAN(pred_real_bottleneck, True)
        )
        loss_D_conv = (loss_D_conv_fake + loss_D_conv_real) * 0.25


        pred_fake_thermal_mamba = self.netD_thermal_mamba(fake_AB)
        pred_real_thermal_mamba = self.netD_thermal_mamba(real_AB)

        loss_D_thermal_mamba_fake = self.criterionGAN(pred_fake_thermal_mamba, False)
        loss_D_thermal_mamba_real = self.criterionGAN(pred_real_thermal_mamba, True)
        loss_D_thermal_mamba = (loss_D_thermal_mamba_fake + loss_D_thermal_mamba_real) * 0.5


        self.loss_D = loss_D_conv + loss_D_thermal_mamba
        self.loss_D.backward()


        self.loss_D_real = ((loss_D_conv_real * 0.25 + loss_D_thermal_mamba_real * 0.5)).detach()
        self.loss_D_fake = ((loss_D_conv_fake * 0.25 + loss_D_thermal_mamba_fake * 0.5)).detach()
        self.loss_D_local_monitor  = loss_D_conv.detach()
        self.loss_D_thermal_mamba_monitor = loss_D_thermal_mamba.detach()

    def backward_G(self):

        fake_B_grad = self.fake_B.clone()
        fake_AB = torch.cat((self.real_A, fake_B_grad), dim=1)


        pred_fake_conv_local, pred_fake_conv_bottleneck = self.netD(fake_AB)
        pred_fake_thermal_mamba = self.netD_thermal_mamba(fake_AB)

        loss_G_GAN_conv = ( self.criterionGAN(pred_fake_conv_local, True)
                         + self.criterionGAN(pred_fake_conv_bottleneck, True)
                          ) * 0.5

        loss_G_GAN_thermal_mamba = self.criterionGAN(pred_fake_thermal_mamba, True)


        self.loss_G_GAN = (loss_G_GAN_conv * 0.8+ loss_G_GAN_thermal_mamba * 0.2)



        real_B_detach = self.real_B.detach()


        self.loss_G_L1 = (self.criterionL1(fake_B_grad, real_B_detach)* self.opt.lambda_A)


        self.loss_ssim = ((1 - self.ssim(fake_B_grad, real_B_detach))* self.opt.lambda_A)


        self.loss_G_SelfPer = (self.criterionSelfPer(fake_B_grad, real_B_detach)* 10.0)
        #self.loss_G_SelfPer = (self.criterionSelfPer(fake_B_grad, real_B_detach)* 1.0)


        self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_L1
                + self.loss_ssim
                +self.loss_G_SelfPer
        )

        self.loss_G.backward()


        self.loss_G_val = self.loss_G.detach()

    def optimize_parameters(self):
        # --- 前向传播 ---
        self.forward()

        # --- 更新 D ---
        for p in self.netD.parameters():
            p.requires_grad = True
        for p in self.netD_thermal_mamba.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        self.optimizer_D_thermal_mamba.zero_grad()
        self.backward_D()

        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netD_thermal_mamba.parameters(), max_norm=1.0)
        self.optimizer_D.step()
        self.optimizer_D_thermal_mamba.step()

        # --- 更新 G（冻结 D）---
        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netD_thermal_mamba.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.backward_G()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optimizer_G.step()

        # --- 恢复 D 的梯度（为下次迭代准备）---
        for p in self.netD.parameters():
            p.requires_grad = True
        for p in self.netD_thermal_mamba.parameters():
            p.requires_grad = True

    def get_current_errors(self):
        from collections import OrderedDict
        errors_ret = OrderedDict([
            ('G_GAN', self.loss_G_GAN),
            ('G_L1', self.loss_G_L1),
            ('G_SelfPer',self.loss_G_SelfPer),
            ('D_real', self.loss_D_real),
            ('D_fake', self.loss_D_fake),
            ('D_Local', self.loss_D_local_monitor),
            ('D_ThermalMamba', self.loss_D_thermal_mamba_monitor)
        ])
        return errors_ret


    @staticmethod
    def get_errors():
        return OrderedDict([
            ('G_GAN', 0), ('G_L1', 0),
            ('G_SelfPer', 0),
            ('D_real', 0), ('D_fake', 0),
            ('D_Local', 0), ('D_ThermalMamba', 0)
        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.thermal_tensor2im(self.fake_B.data)
        real_B = util.thermal_tensor2im(self.real_B.data)

        visual_ret = OrderedDict([
            ('real_A', real_A),
            ('fake_B', fake_B),
            ('real_B', real_B)
        ])

        try:
            decoder = (
                self.netG.module.decoder
                if hasattr(self.netG, 'module')
                else self.netG.decoder
            )
            if hasattr(decoder, 'enhancer_d2'):
                _ = decoder.enhancer_d2.gamma.item()
        except Exception:
            pass

        return visual_ret

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netD_thermal_mamba, 'D_mamba', label, self.gpu_ids)



