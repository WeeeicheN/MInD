import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(current_dir + '/../..'))

import argparse
from trainer import Trainer
from utils import boolstr, get_logger, save_configs, set_seed

#================== MOSI ==================

MOSI_Configs = {
    # Input
    'ncls': 1,
    'max_seq_length': 50,
    ## Video
    'indim_v': 47,
    ## Audio
    'indim_a': 74,
    ## Text
    'indim_t': 768,
}

def get_configs():
    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument('--model', type=str, default='MInD')
    parser.add_argument("--dataset", type=str, default=None, choices=['URFUNNY', 'MOSI', 'MOSEI'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_ids', default='0,1,2,3', type=str, help='multi-gpu ids')
    parser.add_argument('--use_mgpu', type=boolstr, default=False, help='whether to use multi-gpu or not.')
    parser.add_argument('--use_tsboard', type=boolstr, default=False, help='whether to use tensorboard visualization')
    parser.add_argument('--chkpt_dir', default=None, type=str)
    parser.add_argument('--chkpt_name', default=None, type=str)
    parser.add_argument('--save_checkpoint', type=boolstr, default=False, help='whether to save checkpoint or not')
    # ====================================
    # Feature
    ## Backbone
    parser.add_argument('--embdim', type=int, default=256)
    parser.add_argument('--bbtf_vhead', type=int, default=4)
    parser.add_argument('--bbtf_ahead', type=int, default=4)
    parser.add_argument('--bbtf_thead', type=int, default=4)
    parser.add_argument('--bbtf_vlayers', type=int, default=3)
    parser.add_argument('--bbtf_alayers', type=int, default=3)
    parser.add_argument('--bbtf_tlayers', type=int, default=3)
    parser.add_argument('--bertpath', type=str, default='/home/dwc/pretrained/bert-base-uncased')
    ## MInD
    parser.add_argument('--enc_dropout', type=float, default=0.4)
    parser.add_argument('--rev_alpha_type', type=str, default='alpha')
    ## Fusion
    parser.add_argument('--fusedim', type=int, default=128)
    ## Clf
    parser.add_argument('--clf_dropout', type=float, default=0.1)
    ## Activation
    parser.add_argument('--activation', type=str, default=None, choices=['elu', 'gelu', 'relu', 'prelu', 'rrelu', 'leakyrelu','tanh', 'hardtanh'])
    # Train
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--epoch_num", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--rollback_num", type=int, default=1, help="number of rollback trials")
    parser.add_argument("--rollback_patience", type=int, default=5, help="rollback patience")
    # Loss
    parser.add_argument("--info_loss_coeff", type=float, default=1., help="coeff of info loss")
    parser.add_argument("--bt_loss_coeff", type=float, default=1., help="coeff of barlow twins loss")
    parser.add_argument("--hsic_loss_coeff", type=float, default=1., help="coeff of hsic loss")
    parser.add_argument("--rec_loss_coeff", type=float, default=1., help="coeff of reconstruction loss")
    parser.add_argument("--cyr_loss_coeff", type=float, default=1., help="coeff of cyclic rec loss")
    parser.add_argument("--noise_pred_loss_coeff", type=float, default=1., help="coeff of noise pred loss")
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--clip_grad", type=boolstr, default=False, help="whether clip gradient")
    parser.add_argument("--clip_ts", type=float, default=1.0, help="gradient clip threshold")
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'RMSprop'])
    parser.add_argument("--adam_b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--adam_b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="adam: weight decay")
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='explr', choices=['explr', 'plateau'])
    parser.add_argument("--explr_gamma", type=float, default=0.5, help='explr: gamma')
    parser.add_argument("--plateau_patience", type=int, default=10, help='plateau: patience')
    parser.add_argument("--plateau_factor", type=float, default=0.1, help='plateau: factor')

    configs = parser.parse_args()

    return configs

def run():
    configs = get_configs()
    for config in MOSI_Configs.keys():
        setattr(configs, config, MOSI_Configs[config])

    save_dir = configs.chkpt_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = get_logger(configs.model, logger_dir=save_dir)
    logger.info(f"Process ID:{os.getpid()}, System Version:{os.uname()}")
    save_configs(configs, save_dir, logger)

    set_seed(configs.seed)

    trainer = Trainer(configs, logger, save_dir)
    trainer.build()
    trainer.train()

if __name__ == '__main__':
    run()
