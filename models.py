import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from modules import *

class MLPLNAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: bool,
            use_cond: bool,
            activation: str = None,
            cond_channels: int = None,
            condition_bias: float = 0,
            dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        if self.activation is not None:
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'leaky_relu':
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == 'silu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)

            # scale shift first
            x = x * (self.condition_bias + cond)

            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class LatentUNet(nn.Module):
    def __init__(self, T, num_layers=10, dropout=0.1, shape=None, activation='silu',
                 num_time_emb_channels: int = 64, num_time_layers: int = 2):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.shape = shape

        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = shape[-1]
            else:
                a = shape[-1]
                b = shape[-1]
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1:
                layers.append(nn.SiLU())
        self.time_embed = nn.Sequential(*layers)

        self.skip_layers = list(range(1, num_layers))
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = activation
                norm = True
                cond = True
                a, b = shape[-1], shape[-1] * 4
                dropout = dropout
            elif i == num_layers - 1:
                act = None
                norm = False
                cond = False
                a, b = shape[-1] * 4, shape[-1]
                dropout = 0
            else:
                act = 'silu'
                norm = True
                cond = True
                a, b = shape[-1] * 4, shape[-1] * 4
                dropout = dropout

            if i in self.skip_layers:
                a += shape[-1]

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=shape[-1],
                    use_cond=cond,
                    condition_bias=1,
                    dropout=dropout,
                ))

    def forward(self, x, t):
        # Timestep embedding
        t = timestep_embedding(t, self.num_time_emb_channels)
        temb = self.time_embed(t)

        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=temb)
        return h

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class TimeEmbeddingKarras(nn.Module):
    def __init__(self, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()

        self.timembedding = nn.Sequential(
            FourierEmbedding(d_model),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU()
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class AuxiliaryUNetKarras(nn.Module):
    def __init__(self, T, ch=64, ch_mult=None, attn=None, num_res_blocks=2, dropout=0.1, a_dim=32, shape=None):
        super().__init__()
        if attn is None:
            attn = [2]
        if ch_mult is None:
            ch_mult = [1, 2, 4, 8]
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.a_dim = a_dim
        self.time_embedding = TimeEmbeddingKarras(ch, tdim)
        self.fc_a = nn.Linear(self.a_dim, tdim)

        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(AuxResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=True, crossattn=False),
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=False, crossattn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(AuxResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, shape[0], 3, stride=1, padding=1)
        )

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.fc_a.weight)
        init.zeros_(self.fc_a.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, a):
        # Latent embedding
        aemb = self.fc_a(a)

        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb, aemb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            if isinstance(layer, AuxResBlock):
                h = layer(h, temb, aemb)
            else:
                h = layer(h)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, AuxResBlock):
                # for timit remove padding on one side to fit size
                if h.shape[-1] != hs[-1].shape[-1]:
                    h = h[:, :, :, :-(h.shape[-1] - hs[-1].shape[-1])]
                if h.shape[-2] != hs[-1].shape[-2]:
                    h = h[:, :, :-(h.shape[-2] - hs[-1].shape[-2])]
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, aemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

class LatentDiff(nn.Module):
    def __init__(self, args, device, shape, num_layers=10):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.device = device
        steps = args.diffusion_steps_latent
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=args.beta1, end=args.betaT, steps=steps), dim=0).to(device=device)
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=steps).to(device=device)
        self.alphas = 1 - self.betas
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.backbone = LatentUNet(T=steps, num_layers=num_layers, dropout=0.1, shape=shape,
                                       activation='silu')
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, idx=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0),)).to(device=self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            x_tilde = x
        output = self.backbone(x_tilde, idx)

        return (output, epsilon) if get_target else output

class LatentDiffSplit(nn.Module):
    def __init__(self, args, device, shape, num_layers=10):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.device = device
        steps = args.diffusion_steps_latent
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=args.beta1, end=args.betaT, steps=steps), dim=0).to(device=device)
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=steps).to(device=device)
        self.alphas = 1 - self.betas
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.s_dim = args.s_dim
        self.d_dim = args.d_dim
        shape_s = [self.s_dim]
        shape_d = [shape[-1] - self.s_dim]
        self.backbone_s = LatentUNet(T=steps, num_layers=num_layers, dropout=0.1, shape=shape_s,
                                       activation='silu')

        self.backbone_d = LatentUNet(T=steps, num_layers=num_layers, dropout=0.1, shape=shape_d,
                                        activation='silu')
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, idx=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0),)).to(device=self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            x_tilde = x
        x_tilde_s = x_tilde[:, :self.s_dim]
        x_tilde_d = x_tilde[:, self.s_dim:]
        output_s = self.backbone_s(x_tilde_s, idx)
        output_d = self.backbone_d(x_tilde_d, idx)
        output = torch.cat((output_s, output_d), dim=1)

        return (output, epsilon) if get_target else output

class LatentDiffConst(nn.Module):
    def __init__(self, args, device, shape, num_layers=10):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.device = device
        args.beta1 = 0.008
        args.betaT = 0.008
        steps = args.diffusion_steps_latent
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=args.beta1, end=args.betaT, steps=steps), dim=0).to(device=device)
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=steps).to(device=device)
        self.alphas = 1 - self.betas
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.s_dim = args.s_dim
        self.d_dim = args.d_dim
        shape_s = [self.s_dim]
        shape_d = [shape[-1] - self.s_dim]
        self.backbone_s = LatentUNet(T=steps, num_layers=num_layers, dropout=0.1, shape=shape_s,
                                       activation='silu')

        self.backbone_d = LatentUNet(T=steps, num_layers=num_layers, dropout=0.1, shape=shape_d,
                                        activation='silu')
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).abs().mean()
        return loss

    def forward(self, x, idx=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0),)).to(device=self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            x_tilde = x
        x_tilde_s = x_tilde[:, :self.s_dim]
        x_tilde_d = x_tilde[:, self.s_dim:]
        output_s = self.backbone_s(x_tilde_s, idx)
        output_d = self.backbone_d(x_tilde_d, idx)
        output = torch.cat((output_s, output_d), dim=1)

        return (output, epsilon) if get_target else output


class TimePriorEncoder(nn.Module):
    def __init__(self, ch=64, ch_mult=None, attn=None, num_res_blocks=2, dropout=0.1, a_dim=32,
                 hidden_dim=256, s_dim=32, d_dim=32,
                 shape=None, sheared_s=False):
        super().__init__()
        self.sheared_s = sheared_s
        if attn is None:
            attn = [2]
        if ch_mult is None:
            ch_mult = [1, 2, 4, 8, 8]
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        self.d_dim = d_dim
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.shape = shape
        self.a_dim = a_dim
        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock_encoder(
                    in_ch=now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock_encoder(now_ch, now_ch, dropout, attn=True),
            ResBlock_encoder(now_ch, now_ch, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock_encoder(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )

        self.lstm = nn.LSTM(input_size=self.shape[1] * self.shape[2], hidden_size=self.hidden_dim, bidirectional=True,
                            num_layers=1, batch_first=True)


        # motion features from the next RNN
        self.d_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)

        self.fc_s = nn.Linear(self.hidden_dim * 2, self.s_dim)
        self.fc_d = nn.Linear(self.hidden_dim, self.d_dim)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
        init.xavier_uniform_(self.lstm.weight_ih_l0)
        init.xavier_uniform_(self.lstm.weight_hh_l0)
        init.zeros_(self.lstm.bias_ih_l0)
        init.xavier_uniform_(self.fc_s.weight)
        init.zeros_(self.fc_s.bias)
        init.xavier_uniform_(self.fc_d.weight)
        init.zeros_(self.fc_d.bias)

    def forward(self, x):
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        # Downsampling
        h = self.head(x)

        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, ResBlock_encoder):
                h = layer(h)
            else:
                h = layer(h, None, None)  # for downsample module, 0 is placeholder for temb and aemb
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock_encoder):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h)
            else:
                h = layer(h, None, None)  # for upsample module, 0 is placeholder for temb and aemb

        h = torch.flatten(self.tail(h), start_dim=1)
        h_time = h.view(b, v, -1)
        h, _ = self.lstm(h_time)

        ################## EXTRACT STATIC FEATURES #####################

        if self.sheared_s:
            h_backward = h[:, 0, self.hidden_dim:2 * self.hidden_dim]
            h_frontal = h[:, v - 1, 0:self.hidden_dim]
            s = torch.cat((h_frontal, h_backward), dim=1)
            s = self.fc_s(s)
        else:
            s_expand = s = self.fc_s(h)

        ################## EXTRACT DYNAMIC FEATURES #####################
        # pass to one direction rnn
        d, _ = self.d_rnn(h)
        d = self.fc_d(d)
        if self.sheared_s:
            s_expand = s.unsqueeze(1).expand(b, v, self.s_dim)
        a = torch.cat((d, s_expand), dim=2)
        a = a.reshape(b * v, -1)

        assert len(hs) == 0
        return s, d, a

class DiffSDAPriorKarras(nn.Module):
    def __init__(self, args, device, shape, ch_mult=None, attn=None):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -0.4 if args.first_stage_model else -1.2
        self.P_std = 1.0 if args.first_stage_model else 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps
        self.sheared_s = args.sheared_s

        shape = shape[1:]
        self.video_length = args.video_length
        self.shared_noise = args.shared_noise
        self.device = device
        self.d_dim = args.d_dim
        self.s_dim = args.s_dim
        self.hidden_dim = args.hidden_dim
        if ch_mult is None:
            if args.input_size == 28:
                ch_mult = [1, 2, 4, ]
            else:
                ch_mult = [1, 2, 2, 2]
        if attn is None:
            attn = [2]

        self.backbone = AuxiliaryUNetKarras(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels,
                                            a_dim=(args.s_dim + args.d_dim), shape=shape, attn=attn)
        self.encoder = TimePriorEncoder(ch_mult=ch_mult, ch=args.encoder_channels, a_dim=(self.s_dim + self.d_dim),
                                        s_dim=self.s_dim, d_dim=self.d_dim, hidden_dim=self.hidden_dim, shape=shape,
                                        attn=attn, sheared_s=self.sheared_s)


        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}

        output, epsilon, s, d, a, weight = self.forward(x, idx=idx, get_target=True)

        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().cpu().item()

        return loss, to_log

    def forward(self, x, idx=None, a=None, get_target=False):
        b, t, c, h, w = x.shape
        if idx is None:
            rnd_normal = torch.randn([x.shape[0], x.shape[1], 1, 1, 1], device=x.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            weight = weight.view(b * t, 1, 1, 1)
            epsilon = torch.randn_like(x) * sigma
            x_tilde = x + epsilon
            x_tilde = x_tilde.view(b * t, c, h, w)
        else:
            idx = torch.as_tensor([idx for _ in range(x.size(0) * x.size(1))]).to(device=self.device).to(torch.float32)
            sigma = idx
            x_tilde = x.view(b * t, c, h, w)

        c_skip = (self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)).view(b * t, 1, 1, 1)
        c_out = (sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()).view(b * t, 1, 1, 1)
        c_in = (1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()).view(b * t, 1, 1, 1)
        c_noise = (sigma.log() / 4).view(b * t)

        if a is None:
            s, d, a = self.encoder(x)

        output = self.backbone(c_in * x_tilde, c_noise, a)
        output = c_skip * x_tilde + c_out * output

        return (output, epsilon, s, d, a, weight) if get_target else output

class LossWrapper(nn.Module):
    '''
    Wrapper class for loss function to overcome the limitation of torch.nn.parallel.DistributedDataParallel
    which does optimization only on the model forward and our loss function is another function in the model.
    '''

    def __init__(self, model, vq_model=None):
        super().__init__()
        self.vq_model = vq_model
        self.model = model

    def forward(self, args, x, idx=None, curr_epoch=0):
        if not args.latent_dataset and self.vq_model is not None:
            b, v, c, h, w = x.shape
            x = self.vq_model.encode(args, x.view(b*v, c, h, w), idx, curr_epoch).view(b, v, c, h, w)
        return self.model.loss_fn(args, x, idx, curr_epoch)



### TIMIT encoder output: s, d, s_q, d_q, s_mean, s_logvar, d_mean, d_logvar, a, a_q
class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)

class BatchLinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(BatchLinearUnit, self).__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.nrm = nn.BatchNorm1d(out_features)
        self.non_lin = nonlinearity

    def forward(self, x):
        x_lin = self.lin(x)
        x_nrm = self.nrm(x_lin.permute(0, 2, 1))
        return self.non_lin(x_nrm.permute(0, 2, 1))

class TimitEncoder(nn.Module):
    def __init__(self, args, a_dim=32, s_dim=32, d_dim=32, hidden_dim=256, shape=None, num_layers=5, dropout=0.1,
                 no_time=False, mlp_hidden_dim_enc=256):
        super().__init__()
        self.device = args.device
        self.d_dim = d_dim
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.a_dim = a_dim
        self.g_dim = mlp_hidden_dim_enc * 4
        self.mlp_hidden_dim_enc = mlp_hidden_dim_enc
        self.no_time = no_time

        if not self.no_time:
            self.skip_layers = list(range(1, num_layers))
            self.layers = nn.ModuleList([])
            for i in range(num_layers):
                if i == 0:
                    act = 'silu'
                    norm = True
                    cond = False
                    a, b = shape[-1], mlp_hidden_dim_enc * 4
                    dropout = dropout
                elif i == num_layers - 1:
                    act = None
                    norm = True
                    cond = False
                    a, b = mlp_hidden_dim_enc * 4, mlp_hidden_dim_enc * 4
                    dropout = 0
                else:
                    act = 'silu'
                    norm = True
                    cond = False
                    a, b = mlp_hidden_dim_enc * 4, mlp_hidden_dim_enc * 4
                    dropout = dropout

                if i in self.skip_layers:
                    a += shape[-1]

                self.layers.append(
                    MLPLNAct(
                        a,
                        b,
                        norm=norm,
                        activation=act,
                        cond_channels=shape[-1],
                        use_cond=cond,
                        condition_bias=1,
                        dropout=dropout,
                    ))



        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.fc_s = nn.Linear(self.hidden_dim * 2, self.s_dim)
        self.fc_d = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)

        self.initialize()

    def forward(self, x):
        b, v, f = x.shape
        # TODO: UNET like standard architecture
        if not self.no_time:
            h = x
            for i in range(len(self.layers)):
                if i in self.skip_layers:
                    # injecting input into the hidden layers
                    h = torch.cat([h, x], dim=-1)
                h = self.layers[i].forward(x=h)
            x = h

        lstm_out, _ = self.z_lstm(x)

        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, v - 1, 0:self.hidden_dim]
        lstm_out_s = torch.cat((frontal, backward), dim=1)
        s = self.fc_s(lstm_out_s)

        features, _ = self.d_rnn(lstm_out)
        d = self.fc_d(features)

        if self.no_time:
            a = torch.cat((d.view(d.shape[0], -1), s), dim=1)
        else:
            s_expand = s.unsqueeze(1).expand(b, v, self.s_dim)
            a = torch.cat((d, s_expand), dim=-1)

        return s, d, a

    def initialize(self):
        init.xavier_uniform_(self.z_lstm.weight_ih_l0)
        init.xavier_uniform_(self.z_lstm.weight_hh_l0)
        init.zeros_(self.z_lstm.bias_ih_l0)
        init.xavier_uniform_(self.fc_s.weight)
        init.zeros_(self.fc_s.bias)
        init.xavier_uniform_(self.fc_d.weight)
        init.zeros_(self.fc_d.bias)


class TimitDiff(nn.Module):
    def __init__(self, T, num_layers=5, dropout=0.1, shape=None, activation='silu',
                 num_time_emb_channels: int = 64, a_dim=64, mlp_hidden_dim=256):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.shape = shape
        tdim = num_time_emb_channels * 4
        self.a_dim = a_dim
        self.time_embed = TimeEmbeddingKarras(num_time_emb_channels, tdim)
        # self.lstm_a = nn.LSTM(a_dim, tdim * 2, batch_first=True, bias=True,
        #                     bidirectional=False)
        self.fc_a = nn.Linear(a_dim, tdim)


        self.skip_layers = list(range(1, num_layers))
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = activation
                norm = True
                cond = True
                a, b = shape[-1], mlp_hidden_dim * 4
                dropout = dropout
            elif i == num_layers - 1:
                act = None
                norm = False
                cond = False
                a, b = mlp_hidden_dim * 4, shape[-1]
                dropout = 0
            else:
                act = 'silu'
                norm = True
                cond = True
                a, b = mlp_hidden_dim * 4, mlp_hidden_dim * 4
                dropout = dropout

            if i in self.skip_layers:
                a += shape[-1]

            self.layers.append(
                MLPLNActWithCond(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=tdim,
                    use_cond=cond,
                    condition_bias=1,
                    dropout=dropout,
                ))

    def forward(self, x, t, a):
        # Timestep embedding
        b, v = t.shape
        temb = self.time_embed(t.view(-1)).view(b, v, -1)
        # a, _ = self.lstm_a(a)
        a = self.fc_a(a)

        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=-1)
            h = self.layers[i].forward(x=h, t_cond=temb, a_cond=a)
        return h


class MLPLNActWithCond(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_cond: bool,
            norm: bool,
            activation: str = None,
            cond_channels: int = None,
            condition_bias: float = 1,
            dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.use_cond = use_cond
        if self.activation is not None:
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        self.condition_bias = condition_bias

        self.linear = nn.Linear(in_channels, out_channels)
        if self.use_cond:
            self.t_linear_emb = nn.Linear(cond_channels, out_channels*2)
            self.a_linear_emb = nn.Linear(cond_channels, out_channels*2)
            self.t_cond_layers = nn.Sequential(self.act, self.t_linear_emb)
            self.a_cond_layers = nn.Sequential(self.act, self.a_linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'leaky_relu':
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == 'silu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, t_cond, a_cond):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            t_cond = self.t_cond_layers(t_cond)
            a_cond = self.a_cond_layers(a_cond)

            # scale shift first
            scale, shift = torch.chunk(t_cond, 2, dim=-1)
            x = x * (self.condition_bias + scale) + shift

            # then norm
            scale, shift = torch.chunk(a_cond, 2, dim=-1)
            # x = x * (self.condition_bias + a_cond)
            x = x * (self.condition_bias + scale) + shift
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class DiffSDAPriorKarrasTimit(nn.Module):
    def __init__(self, args, device, shape, ch_mult=None):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -0.4
        self.P_std = 1.0
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        shape = shape[1:]
        self.video_length = args.video_length
        self.shared_noise = args.shared_noise
        self.device = device
        self.d_dim = args.d_dim
        self.s_dim = args.s_dim
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp_hidden_dim_enc = args.mlp_hidden_dim_enc
        self.no_time = args.no_time
        ch_mult = [1, 2, 2, 2]
        if self.no_time:
            self.backbone = AuxiliaryUNetKarras(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels,
                                                a_dim=(args.s_dim + args.d_dim * self.video_length), shape=(1, *shape))
        else:
            self.backbone = TimitDiff(T=args.diffusion_steps,
                                            a_dim=(self.d_dim+self.s_dim), shape=shape, mlp_hidden_dim=self.mlp_hidden_dim)
        self.encoder = TimitEncoder(args, a_dim=(self.s_dim + self.d_dim),
                                        s_dim=self.s_dim, d_dim=self.d_dim, hidden_dim=self.hidden_dim,
                                        shape=shape, no_time=self.no_time, mlp_hidden_dim_enc=self.mlp_hidden_dim_enc)




        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        output, epsilon, s, d, a, weight = self.forward(x, idx=idx, get_target=True)

        # denoising matching term
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def forward(self, x, idx=None, a=None, get_target=False):
        b, t, f = x.shape
        if self.no_time:
            shape_A = [b, 1, 1]
            shape_B = [b, t, f]
            shape_C = [b]
            num = b
        else:
            shape_A = [b, t, 1]
            shape_B = [b, t, f]
            shape_C = [b, t]
            num = b * t

        if idx is None:
            rnd_normal = torch.randn(shape_A, device=x.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            weight = weight.view(*shape_A)
            epsilon = torch.randn_like(x) * sigma
            x_tilde = x + epsilon
            x_tilde = x_tilde.view(*shape_B)
        else:
            idx = torch.as_tensor([idx for _ in range(num)]).to(device=self.device).to(torch.float32)
            sigma = idx.view(*shape_A)
            x_tilde = x.view(*shape_B)

        c_skip = (self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)).view(*shape_A)
        c_out = (sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()).view(*shape_A)
        c_in = (1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()).view(*shape_A)
        c_noise = (sigma.log() / 4).view(*shape_C)

        if a is None:
            s, d, a = self.encoder(x)


        output = self.backbone(c_in[:, None] * x_tilde[:, None] if self.no_time else c_in * x_tilde,
                               c_noise, a)
        if self.no_time:
            output = output.squeeze(1)

        output = c_skip * x_tilde + c_out * output

        return (output, epsilon, s, d, a, weight) if get_target else output




class TSEncoder(nn.Module):
    def __init__(self, args, a_dim=32, s_dim=32, d_dim=32, hidden_dim=256, shape=None, num_layers=5, dropout=0.1,
                 no_time=False, mlp_hidden_dim_enc=256):
        super().__init__()
        self.device = args.device
        self.d_dim = d_dim
        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.a_dim = a_dim
        self.g_dim = mlp_hidden_dim_enc * 4
        self.mlp_hidden_dim_enc = mlp_hidden_dim_enc
        self.no_time = no_time

        if not self.no_time:
            self.skip_layers = list(range(1, num_layers))
            self.layers = nn.ModuleList([])
            for i in range(num_layers):
                if i == 0:
                    act = 'silu'
                    norm = True
                    cond = False
                    a, b = shape[-1], mlp_hidden_dim_enc * 4
                    dropout = dropout
                elif i == num_layers - 1:
                    act = None
                    norm = True
                    cond = False
                    a, b = mlp_hidden_dim_enc * 4, mlp_hidden_dim_enc * 4
                    dropout = 0
                else:
                    act = 'silu'
                    norm = True
                    cond = False
                    a, b = mlp_hidden_dim_enc * 4, mlp_hidden_dim_enc * 4
                    dropout = dropout

                if i in self.skip_layers:
                    a += shape[-1]

                self.layers.append(
                    MLPLNAct(
                        a,
                        b,
                        norm=norm,
                        activation=act,
                        cond_channels=shape[-1],
                        use_cond=cond,
                        condition_bias=1,
                        dropout=dropout,
                    ))



        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.fc_s = nn.Linear(self.hidden_dim * 2, self.s_dim)
        self.fc_d = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)

        self.initialize()

    def forward(self, x):
        b, v, f = x.shape
        if not self.no_time:
            h = x
            for i in range(len(self.layers)):
                if i in self.skip_layers:
                    # injecting input into the hidden layers
                    h = torch.cat([h, x], dim=-1)
                h = self.layers[i].forward(x=h)
            x = h

        lstm_out, _ = self.z_lstm(x)

        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, v - 1, 0:self.hidden_dim]
        lstm_out_s = torch.cat((frontal, backward), dim=1)
        s = self.fc_s(lstm_out_s)

        features, _ = self.d_rnn(lstm_out)
        d = self.fc_d(features)

        if self.no_time:
            a = torch.cat((d.view(d.shape[0], -1), s), dim=1)
        else:
            s_expand = s.unsqueeze(1).expand(b, v, self.s_dim)
            a = torch.cat((d, s_expand), dim=-1)

        return s, d, a

    def initialize(self):
        init.xavier_uniform_(self.z_lstm.weight_ih_l0)
        init.xavier_uniform_(self.z_lstm.weight_hh_l0)
        init.zeros_(self.z_lstm.bias_ih_l0)
        init.xavier_uniform_(self.fc_s.weight)
        init.zeros_(self.fc_s.bias)
        init.xavier_uniform_(self.fc_d.weight)
        init.zeros_(self.fc_d.bias)

class DiffSDAPriorKarrasTS(nn.Module):
    def __init__(self, args, device, shape, ch_mult=None):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -0.4
        self.P_std = 1.0
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        shape = shape[1:]
        self.video_length = args.video_length
        self.shared_noise = args.shared_noise
        self.device = device
        self.d_dim = args.d_dim
        self.s_dim = args.s_dim
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.mlp_hidden_dim_enc = args.mlp_hidden_dim_enc
        self.no_time = args.no_time
        ch_mult = [1, 2, 2, 2]
        if self.no_time:
            self.backbone = AuxiliaryUNetKarras(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels,
                                                a_dim=(args.s_dim + args.d_dim * self.video_length), shape=(1, *shape))
        else:
            self.backbone = TimitDiff(T=args.diffusion_steps,
                                            a_dim=(self.d_dim+self.s_dim), shape=shape, mlp_hidden_dim=self.mlp_hidden_dim)
        self.encoder = TSEncoder(args, a_dim=(self.s_dim + self.d_dim),
                                        s_dim=self.s_dim, d_dim=self.d_dim, hidden_dim=self.hidden_dim,
                                        shape=shape, no_time=self.no_time, mlp_hidden_dim_enc=self.mlp_hidden_dim_enc)




        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        x, m_mask, x_len = x
        x = x.repeat(10, 1, 1)  # shape=(M*BS, TL, D)
        if m_mask is not None:
            m_mask = m_mask.repeat(10, 1, 1)
        data = (x, m_mask, x_len)
        output, epsilon, s, d, a, weight = self.forward(data, idx=idx, get_target=True)

        # denoising matching term

        loss = (weight * (output - x).square())
        loss = torch.where(m_mask == 1, torch.zeros_like(loss), loss).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def forward(self, x, idx=None, a=None, get_target=False):
        if isinstance(x, tuple):
            x, mask, x_len = x

        b, t, f = x.shape
        shape_A = [b, t, 1]
        shape_B = [b, t, f]
        shape_C = [b, t]
        num = b * t


        if idx is None:
            rnd_normal = torch.randn(shape_A, device=x.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            weight = weight.view(*shape_A)
            epsilon = torch.randn_like(x) * sigma
            x_tilde = x + epsilon
            x_tilde = x_tilde.view(*shape_B)
        else:
            idx = torch.as_tensor([idx for _ in range(num)]).to(device=self.device).to(torch.float32)
            sigma = idx.view(*shape_A)
            x_tilde = x.view(*shape_B)

        c_skip = (self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)).view(*shape_A)
        c_out = (sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()).view(*shape_A)
        c_in = (1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()).view(*shape_A)
        c_noise = (sigma.log() / 4).view(*shape_C)

        if a is None:
            s, d, a = self.encoder(x)


        output = self.backbone(c_in[:, None] * x_tilde[:, None] if self.no_time else c_in * x_tilde,
                               c_noise, a)
        if self.no_time:
            output = output.squeeze(1)

        output = c_skip * x_tilde + c_out * output

        return (output, epsilon, s, d, a, weight) if get_target else output
