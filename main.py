import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from rhythmnet.inference import RhythmNet
from src.dataset import DFDC, collate_fn
from src.trainer import DFD_Trainer
from src.config import config

def main():
    """ Training and evaluation routines for RhythmNet model. """
    ## Set up environment ##
    # GPU and CUDA settings
    if config.GPU >= 0:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
        torch.cuda.empty_cache()

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # random seed settings
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # results directory
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    # load data
    v_hr_model = RhythmNet()
    train_dl = DataLoader(
        DFDC('train', v_hr_model),
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_dl = DataLoader(
        DFDC('val', v_hr_model),
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False
    )
    test_dl = DataLoader(
        DFDC('test', v_hr_model),
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False
    )

    # train and evaluate model
    trainer = DFD_Trainer(train_dl, val_dl, test_dl)

    print(f"----\nTRAINING & VAL")
    with open(f"{config.RESULTS_PATH}/config_args.txt", 'w') as results_file:
        for arg in vars(config):
            results_file.write(f"{arg:<30}: {getattr(config, arg)}\n")
    trainer.train()

    print(f"----\nTESTING")
    trainer.test()

if __name__ == '__main__':
    main()