import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import PolynomialLR, OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from .models import DFD_Conv1d, DFD_LSTM
from .metrics import compute_metrics, plot_train_val_stats, save_stats
from .config import config

class DFD_Trainer:
    """ Training and testing routines for the V_HR encoder model.
    
    Attributes:
        device (torch.device): Device to be used for training.
        train_dl (torch.utils.data.DataLoader): DataLoader for training set.
        val_dl (torch.utils.data.DataLoader): DataLoader for validation set.
        test_dl (torch.utils.data.DataLoader): DataLoader for test set.
        model (torch.nn.Module): Encoder model for V_HR modality.
        loss (torch.nn.Module): BCE loss function.
        optimiser (torch.optim.Optimiser): Optimiser for model training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        patience (int): Patience counter for early stopping.
        best_val_auroc (float): Best validation AUROC.
        train_val_stats (dict): Dict to store training and validation metrics and losses.
        test_results (dict): Dict to store test metrics.
    """
    def __init__(self, train_dl, val_dl, test_dl):
        self.device = config.DEVICE

        # data loaders
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # model and weights init.
        if config.MODEL == 'conv1d':
            self.model = DFD_Conv1d().to(self.device)
        elif config.MODEL == 'lstm':
            self.model = DFD_LSTM().to(self.device)
        else:
            raise ValueError(f'Invalid model: {config.MODEL}.')

        # BCE loss function, Adam optimiser and LR scheduler
        self.loss = nn.BCELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=config.LR)
        if config.LR_SCH == 'linear':
            self.scheduler = PolynomialLR(self.optimiser, total_iters=config.EPOCHS, power=1)
        elif config.LR_SCH == 'linearW':
            self.scheduler = OneCycleLR(self.optimiser, max_lr=config.LR,
                                        epochs=config.EPOCHS, steps_per_epoch=len(self.train_dl)//config.BATCH_SIZE,
                                        pct_start=0.1, anneal_strategy='linear',
                                        final_div_factor=1.0, three_phase=False)
        elif config.LR_SCH == 'step':
            self.scheduler = ReduceLROnPlateau(self.optimiser, factor=0.5, patience=config.EPOCHS//5, mode='min')
        elif config.LR_SCH == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimiser, T_max=config.EPOCHS)
        elif config.LR_SCH == 'cosine_restart':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimiser, T_0=config.EPOCHS//5)    # make T_0=max_epoch//10 if models converges quickly
            # uniform restarts every max
        else:
            raise ValueError(f'Invalid LR scheduler {config.LR_SCH}.')
        
        # epoch and eval. metrics counters
        self.epoch = 0
        self.patience = 0
        self.best_val_auroc = 0
        self.train_val_stats = {'epoch': [], 'train_acc': [], 'train_auroc': [], 'train_loss': [],
                                'val_acc': [], 'val_auroc': [], 'val_loss': []}
        self.test_results = {}
    
    def train_epoch(self):
        """ Train the model for one epoch on train set. """
        train_preds = torch.FloatTensor().to(self.device)
        train_labels = torch.FloatTensor().to(self.device)
        epoch_loss = 0

        for batch in self.train_dl:
            # load batch data
            v_hr_seqs = batch['V_HR'].to(self.device)
            train_label = batch['label'].to(self.device)

            # forward pass
            train_pred, _ = self.model(v_hr_seqs)
            loss = self.loss(train_pred, train_label)
            epoch_loss += loss.item()

            # backprop and lr step
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            # update learning rate scheduler (if per-batch scheduler is used)
            if isinstance(self.scheduler, (PolynomialLR, OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts)):
                self.scheduler.step()

            # concatenate video samples' predictions and labels across batches
            train_preds = torch.cat((train_preds, train_pred), dim=0)
            train_labels = torch.cat((train_labels, train_label), dim=0)
            
        # epoch end: compute epoch metrics and update self.train_val_stats
        ret = compute_metrics(train_labels.data.cpu().numpy(), train_preds.data.cpu().numpy())
        for metric in ['acc', 'auroc']:
            self.train_val_stats[f'train_{metric}'].append(ret[metric])
        self.train_val_stats['train_loss'].append(epoch_loss / len(self.train_dl))
        self.train_val_stats['epoch'].append(self.epoch)
    
    def val_epoch(self):
        """ Validate the model for one epoch on val set. """
        val_preds = torch.FloatTensor().to(self.device)
        val_labels = torch.FloatTensor().to(self.device)
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.val_dl:
                # load batch data
                v_hr_seqs = batch['V_HR'].to(self.device)
                val_label = batch['label'].to(self.device)

                # forward pass
                val_pred, _ = self.model(v_hr_seqs)
                loss = self.loss(val_pred, val_label)
                epoch_loss += loss.item()

                # concatenate video samples' predictions and labels across batches
                val_preds = torch.cat((val_preds, val_pred), dim=0)
                val_labels = torch.cat((val_labels, val_label), dim=0)
        
        # epoch end: compute epoch metrics and update self.train_val_stats
        ret = compute_metrics(val_labels.data.cpu().numpy(), val_preds.data.cpu().numpy())
        for metric in ['acc', 'auroc']:
            self.train_val_stats[f'val_{metric}'].append(ret[metric])
        self.train_val_stats['val_loss'].append(epoch_loss / len(self.val_dl))
    
    def test_epoch(self):
        """ Test the model on the test set. """
        test_preds = torch.FloatTensor().to(self.device)
        test_labels = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch in tqdm(self.test_dl, total=len(self.test_dl), desc="Batches"):
                # load data from batch
                v_hr_seqs = batch['V_HR'].to(self.device)
                test_label = batch['label'].to(self.device)

                # forward pass
                test_pred, _ = self.model(v_hr_seqs)

                # concatenate video samples' predictions and labels across batches
                test_preds = torch.cat((test_preds, test_pred), dim=0)
                test_labels = torch.cat((test_labels, test_label), dim=0)
        
        # epoch: compute epoch metrics over test set and store in self.test_results
        self.test_results = compute_metrics(test_labels.data.cpu().numpy(), test_preds.data.cpu().numpy())
    
    def train(self):
        """ Training routine for V_HR encoder model. """
        
        for self.epoch in tqdm(range(1, config.EPOCHS + 1), desc="Epochs"):
            # training epoch
            self.model.train()
            self.train_epoch()

            # validation epoch
            self.model.eval()
            self.val_epoch()

            # update learning rate scheduler (if metric-based scheduler)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.train_val_stats['val_loss'][-1])

            # save checkpoint if validation AUROC improves
            if self.train_val_stats['val_auroc'][-1] > self.best_val_auroc:
                self.best_val_auroc = self.train_val_stats['val_auroc'][-1]
                self.save_checkpoint(mode='best')
                self.patience = 0
            else:
                self.patience += 1

            # early stopping
            if self.patience >= config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {self.epoch}")
                break
        
            # update progress bar with this validation epoch's loss and AUROC
            pbar.set_postfix({
                'val_loss': self.train_val_stats['val_loss'][-1],
                'val_auroc': self.train_val_stats['val_auroc'][-1],
            })

        self.save_checkpoint(mode='last')

        # plot and save training and validation metrics per epoch
        plot_train_val_stats(self.train_val_stats)
        save_stats(self.train_val_stats, mode='train_val')

    def test(self):
        """ Testing routine for the V_HR encoder model. """
        # load model from checkpoint, overwriting current model
        self.load_model()

        # test epoch
        self.model.eval()
        self.test_epoch()

        # save and print test metrics
        save_stats(self.test_results, mode='test')
        print(f"Test AUROC: {self.test_results['auroc']:.4g}")
        print(f"Test accuracy: {self.test_results['acc']:.4g}")
        print("----")
    
    def save_checkpoint(self, mode):
        """ Save model weights to a checkpoint .pth file.
        
        Args:
            mode (str): 'best' or 'last' checkpoint to save.
        """
        ckpt_path = os.path.join(config.RESULTS_PATH, f'{mode}_model.pth')
        ckpt = {
            "model": self.model.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def load_model(self):
        """ Load best model weights from checkpoint .pth file. """
        ckpt_path = os.path.join(config.RESULTS_PATH, f'best_model.pth')
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint file found. Train model first.")
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])