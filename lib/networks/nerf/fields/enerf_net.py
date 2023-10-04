import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class ENeRFNet(nn.Module):
    def __init__(self, hid_n=64, feat_ch=8+3, only_rgb=False, **kwargs):
        """
        """
        super(ENeRFNet, self).__init__()
        self.only_rgb = only_rgb
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(16, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.ReLU())
        self.color = nn.Sequential(
                nn.Linear(64+16+feat_ch+4, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, 1),
                nn.ReLU())
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    # def forward(self, vox_feat, img_feat_rgb_dir):
    def forward(self, xyz_encoding, dir_encoding):
        img_feat_rgb_dir_msk = xyz_encoding
        img_feat_rgb_dir = xyz_encoding[..., :-1] # RGB 3, FEAT 8, DIR 4
        B, N_views, N_points = img_feat_rgb_dir.shape[:3]
        
        img_feat = self.agg(img_feat_rgb_dir)
        S = img_feat_rgb_dir.shape[1]
        
        vox_img_feat = img_feat
        x = self.lr0(vox_img_feat)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = None if self.only_rgb else self.sigma(x) 
        x = torch.cat((x, vox_img_feat), dim=-1)
        x = x[:, None].repeat(1, S, 1, 1, 1)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x), dim=1)
        color = torch.sum((img_feat_rgb_dir[..., :3] * color_weight), dim=1)
        return color, sigma

class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if False:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S, N_points, N_samples = img_feat_rgb_dir.shape[:4]
        if False:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]
            
        var_feat = torch.var(img_feat_rgb, dim=1).view(B, 1, N_points, N_samples, self.feat_ch).repeat(1, S, 1, 1, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=1).view(B, 1, N_points, N_samples, self.feat_ch).repeat(1, S, 1, 1, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=1)
        im_feat = (global_feat * agg_w).sum(dim=1)
        return self.fc(im_feat)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)