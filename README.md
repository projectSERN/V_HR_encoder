# V_HR_encoder

This repo contains the code for the V_HR modality's encoder model. Besides the training and test routines for the encoder itself (temporal convolution / LSTM model), this also includes the pre-trained RhythmNet model's inference for first extracting V_HR sequences for the video samples.

### Key points
- V_HR sequences are first generated for all video samples by passing them one-by-one to the optimal pre-trained RhythmNet model (change `config.RHYTHMNET_LOAD_PATH` based on cross-corpus studies' outcomes). These are saved to `v_hr_seqs` folder within `DFDC_subsets/subset_<05/06/07>/<train/val/test>` as .npy files. In subsequent training and testing of encoder model, these sequences are simply loaded from the .npy files instead of passing the videos to RhythmNet again, significantly saving time.

- each DFDC video sample takes about 30s for the combined generation of its ST map and V_HR sequence. Hence, it is important to perform first complete run asap, in order to have all video samples' V_HR sequences generated and ready to be loaded.

- training and evaluation of the encoder model (relevant code in `src/models.py` and `src/trainer.py`) require more attention and its code may need edits. A trial run of `main.py` up to the data loading stage (relevant code in `rhythmnet/inference.py` and `src/dataset.py`) has already been completed, so changes to these files are not expected.

_Note: an optional `run.sh` file has been added to make running `main.py` easier with a short terminal command. This script file can be modified for your specific use case._

_Note: don't forget to populate `rhythmnet/u2net/` with `skin_u2netp.onnx`_