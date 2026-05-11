import torch
import torch.nn as nn
from models.networks import UnetEncoder

class SelfPerceptualLoss(nn.Module):

    def __init__(self, pretrained_path, device, input_nc=3, ngf=64, norm='instance'):
        super(SelfPerceptualLoss, self).__init__()


        print("Forcing ir_encoder initialization to input_nc=1 and norm_layer=nn.BatchNorm2d...")
        self.model = UnetEncoder(input_nc=1, ngf=ngf, norm_layer=nn.BatchNorm2d).to(device)

        #  加载权重文件
        print(f"Loading IR Encoder weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)

        if 'ir_encoder' in checkpoint:
            ir_state_dict = checkpoint['ir_encoder']
        else:
            raise KeyError("The checkpoint does not contain 'ir_encoder'. Check your .pth file.")

        new_state_dict = {}
        for k, v in ir_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)


        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def get_features(self, x):
        feats = []
        d1 = self.model.initial(x)
        feats.append(d1)

        d2 = self.model.down1(d1)
        d3 = self.model.down2(d2)
        feats.append(d3)

        d4 = self.model.down3(d3)
        d5 = self.model.down4(d4)
        feats.append(d5)

        d6 = self.model.down5(d5)
        d7 = self.model.down6(d6)
        feats.append(d7)

        return feats

    def forward(self, fake_B, real_B):

        if fake_B.shape[1] == 3:
            fake_B = fake_B.mean(dim=1, keepdim=True)
        if real_B.shape[1] == 3:
            real_B = real_B.mean(dim=1, keepdim=True)

        feat_fake = self.get_features(fake_B)
        feat_real = self.get_features(real_B)

        loss = 0
        for f, r in zip(feat_fake, feat_real):
            loss += self.criterion(f, r)

        return loss
