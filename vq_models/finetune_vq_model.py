import argparse, os, datetime

import torch
from data import get_dataset


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = True

def get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def nondefault_trainer_args(Trainer, opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['vq8_config.yaml'],
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for log output",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--dataset', required=True,
                        choices=['ffhq', 'mug', 'jester',
                                 'taichi', 'vox1', 'vox2', 'celebv'],
                        help='training dataset')
    parser.add_argument('--video_length', type=int, default=1,
                        help='video length for video dataset.')
    parser.add_argument('--input_size', type=int, default=256,
                        help='expected size of input')
    parser.add_argument('--gpu_num', type=int, default=1, help='gpu number')
    parser.add_argument('--rank', type=int, default=0, help='rank number of process')
    parser.add_argument('--r_seed', type=int, default=0, help='the value of given random seed')

    return parser

def main():
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.trainer import Trainer
    from omegaconf import OmegaConf


    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.r_seed)

    # trainer and callbacks
    trainer_kwargs = dict()

    dataloader = get_dataset(opt)
    evalloader = None
    if isinstance(dataloader, tuple):
        dataloader, evalloader = dataloader

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # cli = OmegaConf.from_dotlist(unknown)
    config = configs[0]
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "cuda"
    for k in nondefault_trainer_args(Trainer, opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    # model
    model = instantiate_from_config(config.model)

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                # "offline": opt.debug,
                "id": nowname,
            }
        },
    }
    tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/')
    # default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    # logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    # trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    # default_modelckpt_cfg = {
    #     "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    #     "params": {
    #         "dirpath": ckptdir,
    #         "filename": "{epoch:06}",
    #         "verbose": True,
    #         "save_last": True,
    #     }
    # }
    # if hasattr(model, "monitor"):
    #     print(f"Monitoring {model.monitor} as checkpoint metric.")
    #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
    #     default_modelckpt_cfg["params"]["save_top_k"] = 3
    #
    # if "modelcheckpoint" in lightning_config:
    #     modelckpt_cfg = lightning_config.modelcheckpoint
    # else:
    #     modelckpt_cfg = OmegaConf.create()
    # modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    # print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    # if version.parse(pl.__version__) < version.parse('1.4.0'):
    #     trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # # add callback which sets up log directory
    # default_callbacks_cfg = {
    #     "setup_callback": {
    #         "target": "main.SetupCallback",
    #         "params": {
    #             "resume": opt.resume,
    #             "now": now,
    #             "logdir": logdir,
    #             "ckptdir": ckptdir,
    #             "cfgdir": cfgdir,
    #             "config": config,
    #             "lightning_config": lightning_config,
    #         }
    #     },
    #     "image_logger": {
    #         "target": "main.ImageLogger",
    #         "params": {
    #             "batch_frequency": 750,
    #             "max_images": 4,
    #             "clamp": True
    #         }
    #     },
    #     "learning_rate_logger": {
    #         "target": "main.LearningRateMonitor",
    #         "params": {
    #             "logging_interval": "step",
    #             # "log_momentum": True
    #         }
    #     },
    #     "cuda_callback": {
    #         "target": "main.CUDACallback"
    #     },
    # }
    # if version.parse(pl.__version__) >= version.parse('1.4.0'):
    #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})
    #
    # if "callbacks" in lightning_config:
    #     callbacks_cfg = lightning_config.callbacks
    # else:
    #     callbacks_cfg = OmegaConf.create()

    # if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
    #     print(
    #         'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
    #     default_metrics_over_trainsteps_ckpt_dict = {
    #         'metrics_over_trainsteps_checkpoint':
    #             {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
    #              'params': {
    #                      "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
    #                  "filename": "{epoch:06}-{step:09}",
    #                  "verbose": True,
    #                  'save_top_k': -1,
    #                  'every_n_train_steps': 10000,
    #                  'save_weights_only': True
    #              }
    #              }
    #     }
    #     default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
    #
    # callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    # if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
    #     callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    # elif 'ignore_keys_callback' in callbacks_cfg:
    #     del callbacks_cfg['ignore_keys_callback']
    #
    # trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, logger=tb_logger, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    # data = instantiate_from_config(config.data)
    # NOTE ac

    # configure learning rate
    bs, base_lr = opt.batch_size, config.model.base_learning_rate
    # if not cpu:
    #     ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    # else:
    ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()