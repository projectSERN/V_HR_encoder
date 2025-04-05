import os
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

from .config import config
    
def compute_metrics(labels, preds):
    """ Compute accuracy and AUROC scores for the given epoch's labels and predictions.
    
    Args:
        labels (np.ndarray): Ground truth labels.
        preds (np.ndarray): Predicted labels.
        
    Returns:
        epoch_metrics (dict): Dict containing accuracy and AUROC scores.

    """
    epoch_metrics = {}

    df = pd.DataFrame({'labels': labels, 'preds': preds})
    acc = accuracy_score(df['labels'], (df['preds']>0.5).astype(int))
    auroc = roc_auc_score(df['labels'], df['preds'])

    epoch_metrics['acc'] = acc
    epoch_metrics['auroc'] = auroc

    return epoch_metrics

def save_stats(stats, mode):
    """ Save training and validation metrics per epoch or test results to a text file.
    
    Args:
        stats (dict): Dictionary containing metrics to save.
        mode (str): 'train_val' or 'test'.
    """
    if mode == 'train_val':
        with open(os.path.join(config.RESULTS_PATH, 'train_val_stats', f'epoch_stats.txt'), 'w') as file:
            file.write('TRAIN AND VAL METRICS PER EPOCH\n\n')
            file.write(f"{'Epoch':<10}{'Train ACC':<15}{'Train AUROC':<15}{'Train Loss':<15}{'Val ACC':<15}{'Val AUROC':<15}{'Val Loss':<15}\n")
            for i in range(len(stats['epoch'])):
                file.write(f"{stats['epoch'][i]:<10}{stats['train_acc'][i]:<15.4g}{stats['train_auroc'][i]:<15.4g}{stats['train_loss'][i]:<15.4g}{stats['val_acc'][i]:<15.4g}{stats['val_auroc'][i]:<15.4g}{stats['val_loss'][i]:<15.4g}\n")

    elif mode == 'test':
        with open(os.path.join(config.RESULTS_PATH, f'test_results.txt'), 'w') as file:
            file.write('TEST RESULTS\n\n')
            file.write(f"ACC: {stats['acc']:.4g}\n")
            file.write(f"AUROC: {stats['auroc']:.4g}\n")

def plot_train_val_stats(train_val_stats):
    plots_path = os.path.join(config.RESULTS_PATH, 'train_val_stats')
    os.makedirs(plots_path, exist_ok=True)
    
    metric_configs = [
        ('ACC', 'Accuracy', train_val_stats['train_acc'], train_val_stats['val_acc'], 'b'),
        ('AUROC', 'AUROC', train_val_stats['train_auroc'], train_val_stats['val_auroc'], 'r'),
        ('Loss', 'Loss', train_val_stats['train_loss'], train_val_stats['val_loss'], 'g')
    ]

    # generate and save individual train+val plots for each metric
    for metric_name, y_label, train_metric, val_metric, colour in metric_configs:
        plt.figure(figsize=(10, 6))
        plt.plot(train_val_stats['epoch'], train_metric, f'{colour}-', label=f'Train {metric_name}')
        plt.plot(train_val_stats['epoch'], val_metric, f'{colour}--', label=f'Val {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_path, f'{metric_name.lower()}.png'))
        plt.close()

    # generate and save a single train+val plot with metrics' subplots
    # accuracy subplot
    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_acc'], 'b-', label='Train ACC')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_acc'], 'b--', label='Val ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # auroc subplot
    plt.subplot(2, 2, 2)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_auroc'], 'r-', label='Train AUROC')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_auroc'], 'r--', label='Val AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(2, 2, 3)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_loss'], 'g-', label='Train Loss')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_loss'], 'g--', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'all_metrics.png'))
    plt.close()