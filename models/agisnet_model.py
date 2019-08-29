import torch
from .base_model import BaseModel
from . import networks
from .vgg import VGG19
import random


class AGISNetModel(BaseModel):
    def name(self):
        return 'AGISNetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_L1', 'G_L1_B', 'G_CX', 'G_CX_B', 'G_GAN', 'G_GAN_B', 'D', 'D_B',
                           'G_L1_val', 'G_L1_B_val', 'local_adv']
        self.loss_G_L1_val = 0.0
        self.loss_G_L1_B_val = 0.0
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # It is up to the direction AtoB or BtoC or AtoC
        self.dirsection = opt.direction

        # AGISNet model only support AtoC now, BtoC and AtoB need to do
        # BicycleGAN model supports all
        assert(self.dirsection == 'AtoC')
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_C', 'real_C_l', 'fake_C']
        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        # D for color
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        # D_B for shape
        use_D_B = opt.isTrain and opt.lambda_GAN_B > 0.0

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, self.opt.nencode, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)

        D_output_nc = (opt.input_nc + opt.output_nc) if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        if use_D_B:
            self.model_names += ['D_B']
            self.netD_B = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD_B, norm=opt.norm, nl=opt.nl,
                                            use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                            gpu_ids=self.gpu_ids)

        # local adversarial loss
        use_local_D = opt.isTrain and opt.lambda_local_D > 0.0
        if use_local_D:
            self.model_names += ['D_local']
            self.netD_local = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD_local, norm=opt.norm, nl=opt.nl,
                                                use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                                num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='none')
            self.criterionL1_reduce = torch.nn.L1Loss()

            # Contextual Loss
            self.criterionCX = networks.CXLoss(sigma=0.5).to(self.device)
            self.vgg19 = VGG19().to(self.device)
            self.vgg19.load_model(self.opt.vgg)
            self.vgg19.eval()
            self.vgg_layers = ['conv3_3', 'conv4_2']

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

            if use_D_B:
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_B)

            if use_local_D:
                self.optimizer_Dlocal = torch.optim.Adam(self.netD_local.parameters(), lr=opt.lr,
                                                         betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Dlocal)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)  # A is the base font
        self.real_B = input['B'].to(self.device)  # B is the gray shape
        self.real_B_G = input['B_G'].to(self.device)  # B_G is for GAN, label == 1: B_G == B, else B_G == Shapes[rand]
        self.real_C = input['C'].to(self.device)  # C is the color font
        self.real_C_G = input['B_G'].to(self.device)  # C_G is for GAN, label == 1: C_G == C, else C_G == Colors[rand]
        self.real_C_l = input['C_l'].to(self.device)  # C_l is the C * label, useful to visual
        self.label = input['label'].to(self.device)  # label == 1 means the image is in the few set

        self.real_Bases = input['Bases'].to(self.device)
        self.real_Shapes = input['Shapes'].to(self.device)
        self.real_Colors = input['Colors'].to(self.device)  # Colors is reference color characters

        self.blur_Colors = input['blur_Colors'].to(self.device)

    def test(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            test_l1_loss = torch.nn.functional.l1_loss(self.fake_C, self.real_C)
            return self.real_A, self.fake_B, self.real_B, self.fake_C, self.real_C, test_l1_loss

    def validate(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            self.loss_G_L1_val = torch.nn.functional.l1_loss(self.fake_C, self.real_C)
            self.loss_G_L1_B_val = torch.nn.functional.l1_loss(self.fake_B, self.real_B)
            return self.real_A, self.fake_B, self.real_B, self.fake_C, self.real_C, \
                self.loss_G_L1_B_val, self.loss_G_L1_val

    def train(self):
        for name in self.model_names:
            model_name = 'net' + name
            getattr(self, model_name).train()

    def forward(self):
        # generate fake_C
        self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
        # vgg
        self.vgg_fake_C = self.vgg19(self.fake_C)
        self.vgg_real_C = self.vgg19(self.real_C)
        self.vgg_fake_B = self.vgg19(self.fake_B)
        self.vgg_real_B = self.vgg19(self.real_B)

        # local blocks
        self.fake_C_blocks, self.real_color_blocks, self.blur_color_blocks = \
            self.generate_random_block(self.fake_C, self.real_Colors, self.blur_Colors)

        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_B = torch.cat([self.real_A, self.fake_B], 1)
            self.real_data_B = torch.cat([self.real_A, self.real_B_G], 1)
            self.fake_data_C = torch.cat([self.real_A, self.fake_C], 1)
            self.real_data_C = torch.cat([self.real_A, self.real_C_G], 1)
        else:
            self.fake_data_B = self.fake_B
            self.real_data_B = self.real_B_G
            self.fake_data_C = self.fake_C
            self.real_data_C = self.real_C_G

    def backward_D(self, netD, real, fake, blur=None):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        # blur
        loss_D_blur = 0.0
        if blur is not None:
            pred_blur = netD(blur)
            loss_D_blur, _ = self.criterionGAN(pred_blur, False)

        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real + loss_D_blur
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real, loss_D_blur]

    def backward_G_GAN(self, fake, net=None, ll=0.0):
        if ll > 0.0:
            pred_fake = net(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1. G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_C, self.netD, self.opt.lambda_GAN)
        self.loss_G_GAN_B = self.backward_G_GAN(self.fake_data_B, self.netD_B, self.opt.lambda_GAN_B)

        # 2. reconstruction |fake_C-real_C| |fake_B-real_B|
        self.loss_G_L1 = 0.0
        self.loss_G_L1_B = 0.0
        if self.opt.lambda_L1 > 0.0:
            L1 = torch.mean(torch.mean(torch.mean(
                self.criterionL1(self.fake_C, self.real_C), dim=1), dim=1), dim=1)
            self.loss_G_L1 = torch.sum(self.label * L1) * self.opt.lambda_L1
            L1_B = torch.mean(torch.mean(torch.mean(
                self.criterionL1(self.fake_B, self.real_B), dim=1), dim=1), dim=1)
            self.loss_G_L1_B = torch.sum(self.label * L1_B) * self.opt.lambda_L1_B

        # 3. contextual loss
        self.loss_G_CX = 0.0
        self.loss_G_CX_B = 0.0
        if self.opt.lambda_CX > 0.0:
            for l in self.vgg_layers:
                # symmetric contextual loss
                cx, cx_batch = self.criterionCX(self.vgg_real_C[l], self.vgg_fake_C[l])
                self.loss_G_CX += torch.sum(cx_batch * self.label) * self.opt.lambda_CX
                cx_B, cx_B_batch = self.criterionCX(self.vgg_real_B[l], self.vgg_fake_B[l])
                self.loss_G_CX_B += torch.sum(cx_B_batch * self.label) * self.opt.lambda_CX_B

        # 4. local adv loss
        self.loss_local_adv = 0.0
        if self.opt.lambda_local_D > 0.0:
            self.loss_local_adv = self.backward_G_GAN(self.fake_C_blocks, self.netD_local, self.opt.lambda_local_D)

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_B \
            + self.loss_G_L1 + self.loss_G_L1_B \
            + self.loss_G_CX + self.loss_G_CX_B \
            + self.loss_local_adv

        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        # update D
        if self.opt.lambda_GAN > 0.0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_C, self.fake_data_C)
            self.optimizer_D.step()

        if self.opt.lambda_GAN_B > 0.0:
            self.set_requires_grad(self.netD_B, True)
            self.optimizer_D_B.zero_grad()
            self.loss_D_B, self.losses_D_B = self.backward_D(self.netD_B, self.real_data_B, self.fake_data_B)
            self.optimizer_D_B.step()

        if self.opt.lambda_local_D > 0.0:
            self.set_requires_grad(self.netD_local, True)
            self.optimizer_Dlocal.zero_grad()
            self.loss_Dlocal, self.losses_Dlocal = self.backward_D(self.netD_local, self.real_color_blocks,
                                                                   self.fake_C_blocks, self.blur_color_blocks)
            self.optimizer_Dlocal.step()

    def update_G(self):
        # update dual net G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_B, False)
        # self.set_requires_grad(self.netD_local, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()

    def pretrain_D_local(self):
        self.forward()
        self.update_D()

    def generate_random_block(self, input, target, blurs):
        batch_size, channel, height, width = target.size()  # B X 3*nencode X 64 X 64
        target_tensor = target.data
        block_size = self.opt.block_size
        img_size = self.opt.fineSize  # 64

        inp_blk = list()
        tar_blk = list()
        blr_blk = list()

        for b in range(batch_size):
            for i in range(self.opt.block_num):
                rand_idx = random.randint(0, self.opt.nencode-1)
                x = random.randint(0, height - block_size - 1)
                y = random.randint(0, width - block_size - 1)
                target_random_block = torch.tensor(target_tensor[b, rand_idx*3:(rand_idx+1)*3,
                                                   x:x + block_size, y:y + block_size], requires_grad=False)
                target_blur_block = torch.tensor(blurs[b, rand_idx*3:(rand_idx+1)*3,
                                                 x:x + block_size, y:y + block_size], requires_grad=False)
                if i == 0:
                    target_blocks = target_random_block
                    blur_blocks = target_blur_block
                else:
                    target_blocks = torch.cat([target_blocks, target_random_block], 0)
                    blur_blocks = torch.cat([blur_blocks, target_random_block], 0)

                """
                    x_m = random.randint(0, width-block_size-1)
                    y_m = random.randint(0, height-block_size-1)
                    input_blocks.append(torch.tensor(input.data[:, x_m:x_m+block_size, y_m:y_m+block_size].unsqueeze(0),
                                                     requires_grad=False))
                """
                x1 = random.randint(0, img_size - block_size)
                y1 = random.randint(0, img_size - block_size)
                input_random_block = torch.tensor(input.data[b, :, x1:x1 + block_size, y1:y1 + block_size],
                                                  requires_grad=False)
                if i == 0:
                    input_blocks = input_random_block
                else:
                    input_blocks = torch.cat([input_blocks, target_random_block], 0)

            input_blocks = torch.unsqueeze(input_blocks, 0)
            target_blocks = torch.unsqueeze(target_blocks, 0)
            blur_blocks = torch.unsqueeze(blur_blocks, 0)
            inp_blk.append(input_blocks)
            tar_blk.append(target_blocks)
            blr_blk.append(blur_blocks)

        inp_blk = torch.cat(inp_blk)
        tar_blk = torch.cat(tar_blk)
        blr_blk = torch.cat(blr_blk)

        return inp_blk, tar_blk, blr_blk
