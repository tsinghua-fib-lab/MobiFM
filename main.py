import argparse
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # cuda
from models_with_mask_scale_multiprompt_moe_memory import DiT_models
from train import TrainLoop
import setproctitle
import torch
from DataLoader import data_load_main
from utils import *
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusion import create_diffusion
import datetime



def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        lr=1e-3,
        task = 'mix',
        early_stop = 20,
        weight_decay=1e-4,
        log_interval=20,
        num_memory = 128,
        batch_size = 64,
        total_epoches = 200,
        device_id = 0,
        lr_anneal_steps = 200,
        clip_grad = 0.5,
        mask_strategy = {'generation_masking':[1],'short_long_temporal_masking':[0.25,0.75],'random_masking':[0.75]},
        min_lr = 1e-5,
        dataset = 'NCUGM30_RSRP1',#NCUGM30_NCTCM30_NNTGH_NNUGH_NJUCH_NJTCH_RSRP1_RSRP2
        sample=1,
        t_patch_size=1,
        patch_size =1,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.dataset, args.total_epoches))
    setup_init(88)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.folder = '{}/'.format('multiData_' + current_time)
    args.datatype = args.dataset
    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)


    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    print('start data load')

    data, val_data, test_data, args.scaler = data_load_main(args)

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    model = DiT_models['DiT-S/8'](
        args=args,
        depth=6,
        hidden_size=128,
        patch_size=args.t_patch_size,

    ).to(device)
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=500)

    total_params = sum(p.numel() for p in model.parameters())
    print("Model Para.:", total_params)


    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        diffusion = diffusion,
        data=data,
        test_data=test_data,
        val_data=val_data,
        device=device
    ).run_loop(args)

    print('Model_path', args.folder)

if __name__ == "__main__":
    main()