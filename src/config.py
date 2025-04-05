import os
import argparse

""" Command line arguments for the entire repo. """

parser = argparse.ArgumentParser()

# for main.py
parser.add_argument('--GPU', type=int, default=0, help='ID of GPU to use')
parser.add_argument('--SEED', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size')

# for trainer.py
parser.add_argument('MODEL', type=str, choices=['conv1d', 'lstm'], help='Temporal convolution or LSTM model for deepfake detection using V_HR modality')
parser.add_argument('--LR', type=float, default=0.0001, help='(Max) Learning rate')
parser.add_argument('--LR_SCH', type=str, default='step', help='Learning rate scheduler, default is step/ReduceLROnPlateau')
parser.add_argument('--EPOCHS', type=int, default=20, help='Max number of epochs for training routine')
parser.add_argument('--PATIENCE', type=int, default=5, help='Patience for early stopping')

# for models.py
parser.add_argument('--NUM_BLOCKS', type=int, default=3, help='Number of conv blocks in the Conv1d model')
parser.add_argument('--KERNEL_SIZE', type=int, default=3, help='Kernel size for Conv1d layers of Conv1d model')
parser.add_argument('--POOLING_SIZE', type=int, default=1, help='Kernel size for MaxPool1d layers of Conv1d model')
parser.add_argument('--NUM_LAYERS', type=int, default=1, help='Number of LSTM layers in the LSTM model')
parser.add_argument('--HIDDEN_SIZE', type=int, default=256, help='Hidden size for LSTM layers of LSTM model')
parser.add_argument('--DROPOUT_RATE', type=float, default=0.0, help='Dropout rate for Conv1d and LSTM models')

# for inference.py and dataset.py
parser.add_argument('RHYTHMNET_LOAD_PATH', type=str, help='Path to the optimal pre-trained RhythmNet model\'s weights')
parser.add_argument('DFDC_PATH', type=str, help='Path to the DFDC_subsets directory')
parser.add_argument('--FRAME_RATE', type=int, default=29, help='Frame rate of the DFDC video samples')
parser.add_argument('--CLIP_DUR', type=float, default=5, help='Clip size in seconds')
parser.add_argument('--STRIDE_DUR', type=float, default=0.4, help='Stride length in seconds')
parser.add_argument('--ROI_GRID_SIZE', type=int, default=5, help='Side length of ROI cells for RhythmNet\'s ST map generation')
parser.add_argument('--VERBOSE', action='store_true', help='Verbose mode for RhythmNet\'s ST map generation')

config = parser.parse_args()

# clip size and stride length in frames
config.CLIP_SIZE = int(config.FRAME_RATE * config.CLIP_DUR)  # for 29fps videos, 5s = 145 frames
config.STRIDE = int(config.FRAME_RATE * config.STRIDE_DUR)   # for 29fps videos, 0.4s = 12 frames

config.RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'