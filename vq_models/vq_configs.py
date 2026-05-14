import os
from vq_models.vq import VQModelInterface, AutoencoderKL


model_paths = {'vq8': 'vq-f8-n256_model.ckpt',
               'vq4': 'vqf4_model.ckpt',
               'kl8': 'kl-f8.ckpt',
               'vq8ft': 'vq-f8-n256_model_ft.ckpt'
               }
model_class = {
    'vq8': VQModelInterface,
    'vq8ft': VQModelInterface,
    'vq4': VQModelInterface,
    'kl8': AutoencoderKL,
}

ddconfigs = {
    'vq8': {
        'double_z': False,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [32],
        'dropout': 0.0},
    'vq8ft': {
        'double_z': False,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [32],
        'dropout': 0.0},
    'vq4': {
        'double_z': False,
        'z_channels': 3,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
    },
    'kl8': {
        'double_z': True,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
    }
}
n_embeds = {
    'vq8ft': 256,
    'vq8': 256,
    'vq4': 8192,
    'kl8': None,
}

embed_dims = {
    'vq8ft': 4,
    'vq8': 4,
    'vq4': 3,
    'kl8': 4,
}

lossconfig = {
    'target': 'torch.nn.Identity',
}


def get_fs_model(args, device):
    from paths import PRETRAINED_ROOT
    root = PRETRAINED_ROOT
    path = f'{root}/{model_paths[args.first_stage_model]}'
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist please download the weights')
    ddconfig = ddconfigs[args.first_stage_model]
    model = (model_class[args.first_stage_model](args, embed_dims[args.first_stage_model], ddconfig,
                                                lossconfig, n_embed=n_embeds[args.first_stage_model], ckpt_path=path)
             .to(device=device))
    model.eval()
    args.vq_input_size = args.input_size // (2 ** (len(ddconfig['ch_mult'])-1))
    args.z_channels_vq = ddconfig['z_channels']
    for param in model.parameters():
        param.requires_grad = False
    return model
