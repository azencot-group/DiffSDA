import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

from paths import CLASSIFIERS_ROOT


# ---------------- encoder -----------------------
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


# ---------------- decoder -----------------------
"""
# Using transpose conv as the block to up-sample
"""
class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class decoder_convT(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_convT, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


class decoder_convT_static(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_convT_static, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], output.shape[1], output.shape[2], output.shape[3])

        return output


"""
# Using bilinear upsampling and conv as the block to up-sample
"""

class upconv(nn.Module):
    def __init__(self, nc_in, nc_out):
        super().__init__()
        self.conv = nn.Conv2d(nc_in, nc_out, 3, 1, 1)
        self.norm = nn.BatchNorm2d(nc_out)

    def forward(self, input):
        out = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        return F.relu(self.norm(self.conv(out)))

class decoder_conv(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_conv, self).__init__()
        self.dim = dim
        nf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
            # state size. (nf*8) x 4 x 4
            upconv(nf * 8, nf * 4),
            # state size. (nf*4) x 8 x 8
            upconv(nf * 4, nf * 2),
            # state size. (nf*2) x 16 x 16
            upconv(nf * 2, nf * 2),
            # state size. (nf*2) x 32 x 32
            upconv(nf * 2, nf),
            # state size. (nf) x 64 x 64
            nn.Conv2d(nf, nc, 1, 1, 0),
            nn.Sigmoid()
        )


    def forward(self, input):
        output = self.main(input.view(-1, self.dim, 1, 1))
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output

class decoder_conv_static(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_conv_static, self).__init__()
        self.dim = dim
        nf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
            # state size. (nf*8) x 4 x 4
            upconv(nf * 8, nf * 4),
            # state size. (nf*4) x 8 x 8
            upconv(nf * 4, nf * 2),
            # state size. (nf*2) x 16 x 16
            upconv(nf * 2, nf * 2),
            # state size. (nf*2) x 32 x 32
            upconv(nf * 2, nf),
            # state size. (nf) x 64 x 64
            nn.Conv2d(nf, nc, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.dim, 1, 1))
        output = output.view(input.shape[0], output.shape[1], output.shape[2], output.shape[3])

        return output




class classifier_MUG(nn.Module):
    def __init__(self):
        super(classifier_MUG, self).__init__()
        self.g_dim = 128  # frame feature
        self.channels = 3  # frame feature
        self.hidden_dim = 256
        self.frames = 15
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_dyn = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_st = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 52))

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_dyn(lstm_out_f), self.cls_st(lstm_out_f)



def get_classifier():
    classifier = classifier_MUG()
    cls_path = os.path.join(CLASSIFIERS_ROOT, 'mug_cls_new_contrastive.tar')
    loaded_dict = torch.load(cls_path, map_location='cpu')
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.eval()
    return classifier

