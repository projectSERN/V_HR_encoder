import sys
import os
import cv2
import ssl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from sklearn import preprocessing

from .rtgene.extract_landmarks_method_base import LandmarkMethodBase
from .u2net.inference import U2Net
from src.config import config

ssl._create_default_https_context = ssl._create_stdlib_context

class RhythmNet(nn.Module):
    """ RhythmNet pipeline for generating video sample's ST map and predicting its V_HR sequence at inference time.
    
    Attributes:
        detection_model (LandmarkMethodBase): RT-GENE model for face detection and landmark estimation.
        skin_seg_model (U2Net): U2-Net model for skin segmentation.
        min_max_norm (MinMaxScaler): MinMaxScaler for normalising ST maps.
        resnet18 (nn.Sequential): RhythmNet's CNN model (pre-trained ResNet-18).
        linear (nn.Linear): Linear projection layer.
        batch_norm1 (nn.BatchNorm1d): Batch normalisation layer 1.
        batch_norm2 (nn.BatchNorm1d): Batch normalisation layer 2.
        dropout (nn.Dropout): Dropout layer.
        regression (nn.Linear): Final regression layer.
    """
    def __init__(self):
        super(RhythmNet, self).__init__()

        # initialise RT-GENE model (for face detection and landmark estimation) and U2-Net model (for skin segmentation)
        src_path = os.path.dirname(os.path.realpath(__file__))
        self.detection_model = LandmarkMethodBase(
            device_id_facedetection=config.DEVICE,
            checkpoint_path_face=os.path.join(src_path, "rtgene/model_nets/SFD/s3fd_facedetector.pth"),
            checkpoint_path_landmark=os.path.join(src_path, "rtgene/model_nets/phase1_wpdc_vdc.pth.tar"),
            model_points_file=os.path.join(src_path, "rtgene/model_nets/face_model_68.txt")
        )
        self.skin_seg_model = U2Net(model_path=os.path.join(src_path, "u2net", "skin_u2netp.onnx"))

        # initialise MinMaxScaler for normalising ST maps
        self.min_max_norm = preprocessing.MinMaxScaler()

        # initialise RhythmNet's CNN model (pre-trained ResNet-18)
        resnet = models.resnet18(weights='DEFAULT')
        modules = list(resnet.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)                             # remove ResNet-18's classifier layer
        
        # linear projection and regularisation layers
        self.linear = nn.Linear(512, 1000)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(0.3)

        # final regression layer
        self.regression = nn.Linear(1000, 1)

        # move model to device
        self.to(config.DEVICE)
        
        # load optimal pre-trained model weights
        self.load_weights(config.RHYTHMNET_LOAD_PATH)

    def load_weights(self, ckpt_path):
        """ Load pre-trained weights for the RhythmNet model.

        Args:
            weights_path (str): Path to the pre-trained weights .pth file.
        """
        state_dict = torch.load(ckpt_path, map_location=config.DEVICE)['model']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("rnn.")}  # ignore GRU weights as GRU is not used in inference
        self.load_state_dict(state_dict)
        print("----\nEXTRACT V_HR MODALITY VIA RhythmNet")
        print("Loaded pre-trained weights successfully")

    def forward(self, video_path):
        """ Feed a video sample to the pre-trained RhythmNet model to get its V_HR sequence.
        
        Args:
            video_path (str): Path to the video sample.

        Returns:
            clip_preds (torch.Tensor): Predicted V_HR sequence for the video sample, shape: (num_clips,).
        """
        frame_preds = []
        frame_feats = []

        # 1/2: generate ST maps for the video sample (or load them if they were already generated) #
        st_maps = self.process_video(video_path)                            # shape: (num_clips, T, n, c)
        if st_maps is None:
            # if no >=5s of continuous frames with detected face were found, return zero tensor of shape (1,)
            return torch.zeros(1).to(config.DEVICE).float()
        st_maps = torch.from_numpy(st_maps).to(config.DEVICE).float()       # shape: (num_clips, T, n, c)

        # 2/2: generate video sample's V_HR sequence #
        for t in range(st_maps.size(1)):                                    # st_maps shape: (num_clips, T, n, c)
            # pass ST maps of all clips in video at frame t through ResNet-18
            x = (st_maps[:, t, :, :].permute(0, 2, 1)).unsqueeze(-1)        # shape: (num_clips, c, n, 1)
            x = self.resnet18(x)                                            # shape: (num_clips, 512, 1, 1)

            # collapse frame-wise features to 2d tensor
            x = x.view(x.size(0), -1)                                       # shape: (num_clips, 512)

            # regularisation and linear projection of frame-wise features
            x = self.batch_norm1(x)
            x = self.dropout(x)
            x = F.relu(self.linear(x))                                      # shape: (num_clips, 1000)
            x = self.batch_norm2(x)
            x = self.dropout(x)
            frame_feats.append(x)

            # get frame-wise HR values
            x = self.regression(x)                                          # shape: (num_clips, 1)
            x = x * config.FRAME_RATE
            frame_preds.append(x.squeeze(-1))

        # average frame-wise HR values to get clip-wise HR values
        clip_preds = torch.stack(frame_preds, dim=0).mean(dim=0)            # shape after stacking: (T, num_clips), shape after mean: (num_clips,)
        return clip_preds                                                   # shape: (num_clips,)

    def get_frames(self, video_path):
        """ Extract frames of a video.

        Args:
            video_path (str): Path to the video sample.

        Returns:
            frames (np.ndarray): Array of frames of the video.
        """
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(7))
        frames = np.zeros((num_frames, int(cap.get(4)), int(cap.get(3)), 3), dtype='uint8') # shape: (num_frames, H, W, C)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if (not ret) and (frame_idx < num_frames):                                      # if frame could not be read
                if config.VERBOSE:
                    print(f"{os.path.basename(video_path)}: frame {frame_idx} replaced with black frame as it could not be read")
                frames[frame_idx, :, :, :] = np.full_like(frames[frame_idx, :, :, :], 0)
                frame_idx += 1

            elif (not ret) and (frame_idx == num_frames):                                   # stop once all frames have been read
                break

            else:
                frames[frame_idx, :, :, :] = frame
                frame_idx += 1

        cap.release()
        return frames
    
    def process_frame(self, frame, detection_model, skin_seg_model):
        """ For a video frame: (1) detect face area and facial landmarks, (2) crop face area, (3) remove non-skin regions, (4) remove eye region, and (5) convert to YUV.

        Args:
            frame (np.ndarray): Video frame to get face area from.
            detection_model (LandmarkMethodBase): RT-GENE model for face detection and landmark estimation.
            skin_seg_model (U2Net): U2-Net model for skin segmentation.

        Returns:
            processed_frame (np.ndarray): Frame's face area in YUV colour space with skin segmented and eye region removed.
            face_det_flag (bool): Flag indicating if a face was detected in the frame.
        """
        # (1) detect face area and facial landmarks
        faceboxes = detection_model.get_face_bb(frame)
        if len(faceboxes) == 0:
            return None, False                                                              # if no face detected in the frame
        
        subjects = detection_model.get_subjects_from_faceboxes(frame, [faceboxes[0]])
        if len(subjects) == 0:
            return None, False                                                              # if no facial landmarks could be estimated for this face
        
        subject = subjects[0]                                                               # first / most prominent face in the frame
        if hasattr(subject, 'box'):
            # (2) crop face area
            x1, y1, x2, y2 = map(int, subject.box)
            face_area = frame[y1:y2, x1:x2]

            # (3) apply U2Net skin segmentation model to remove non-skin regions
            segmented_face = skin_seg_model.predict(face_area)

            # (4) if relevant facial landmarks are available, remove eye region
            if hasattr(subject, 'landmarks') and subject.landmarks is not None:
                left_eye = np.mean(subject.landmarks[36:42], axis=0).astype(int)
                right_eye = np.mean(subject.landmarks[42:48], axis=0).astype(int)

                # eye position relative to face region
                le_x, le_y = left_eye[0] - x1, left_eye[1] - y1
                re_x, re_y = right_eye[0] - x1, right_eye[1] - y1

                # eye region to be masked out
                eye_radius = int((x2-x1) * 0.05)
                cv2.circle(segmented_face, (le_x, le_y), eye_radius, (0, 0, 0), -1)
                cv2.circle(segmented_face, (re_x, re_y), eye_radius, (0, 0, 0), -1)

            # (5) convert to YUV
            segmented_face_yuv = cv2.cvtColor(segmented_face, cv2.COLOR_BGR2YUV)
            
            return segmented_face_yuv, True
        
        return None, False

    def select_frames(self, face_det_flags):
        """ Select >=5s of continuous frames with detected face to handle cases where no face was detected in a certain frame.

        Args:
            face_det_flags (list): List of flags indicating if a face was detected in the corresponding frame.

        Returns:
            (list): List of indices of continuous frames with detected face.
        """
        # iterate over frames to segment continuous frames with detected face
        counter = 0
        valid_frames = {}
        invalid_frames = []

        valid_frames[counter] = []
        for frame_idx, flag_det_flag in enumerate(face_det_flags):
            if flag_det_flag:
                if len(invalid_frames) > 0:
                    if len(invalid_frames) >= config.FRAME_RATE:
                        counter += 1
                        valid_frames[counter] = [frame_idx]
                    else:
                        valid_frames[counter].append(frame_idx)
                    invalid_frames = []
                else:
                    valid_frames[counter].append(frame_idx)

            else:
                invalid_frames.append(frame_idx)

        # determine if >=5s of continuous frames with detected face are available and return them
        if counter == 0:
            if len(valid_frames[0]) != 0:
                return valid_frames[0]                                                          # cases 1 and 2: entire video is valid or video is interrupted by <1s of invalid frames (these are ignored)
            else:
                return None                                                                     # case 5: video has no detected face in any frame
        else:
            max_len_key = 0
            for k in valid_frames.keys():
                if len(valid_frames[k]) > len(valid_frames[max_len_key]):
                    max_len_key = k

            if len(valid_frames[max_len_key]) >= 5 * config.FRAME_RATE:
                return valid_frames[max_len_key]                                                # case 3: video is interrupted by >1s of invalid frames but at least 5s of uninterrupted valid frames found
            else:
                return None                                                                     # case 4: video is interrupted by >1s of invalid frames and not even 5s of uninterrupted valid frames found

    def get_roi_cells(self, face_area):
        """ Get ROIs in a frame's face area by dividing it into a 5 x 5 grid.

        Args:
            face_area (np.ndarray): Face area to identify ROIs in.

        Returns:
            roi_cells (list): Left-to-right, top-to-bottom list of ROIs in the face area.
        """
        roi_cells = []
        vert_size = face_area.shape[0]
        horiz_size = face_area.shape[1]

        # indices for dividing the face area into a 5 x 5 grid
        vert_interval = vert_size // config.ROI_GRID_SIZE
        horiz_interval = horiz_size // config.ROI_GRID_SIZE
        vert_indices = [i for i in range(0, face_area.shape[0] + 1, vert_interval)]
        horiz_indices = [i for i in range(0, face_area.shape[1] + 1, horiz_interval)]

        # select ROIs using the indices, going from left to right, top to bottom
        for i in range(len(vert_indices) - 1):                                                  # iterate over rows or height
            vert_start = vert_indices[i]       
            vert_end = vert_indices[i + 1]

            for j in range(len(horiz_indices) - 1):                                             # iterate over columns or width
                horiz_start = horiz_indices[j]
                horiz_end = horiz_indices[j + 1]

                roi_cells.append(face_area[vert_start:vert_end, horiz_start:horiz_end])

        return roi_cells                                                                        # len: 25, each element an array of shape: (144, 192)

    def process_video(self, video_path):
        """ Split video sample into overlapping clips and generate spatio-temporal maps for each clip.

        Args:
            video_path (str): Path to the video sample.
        """

        # get frames from video
        frames = self.get_frames(video_path)
        if config.VERBOSE:
            print(f"{os.path.basename(video_path)}: read {frames.shape[0]} frames")

        # process each frame to get processed face area
        processed_frames = []
        face_det_flags = []
        for frame in frames:
            processed_frame, face_det_flag = self.process_frame(frame, self.detection_model, self.skin_seg_model)

            if face_det_flag:                                                                   # if face and landmarks detected, append processed frame
                processed_frames.append(processed_frame)
                face_det_flags.append(True)
            else:                                                                               # if no face detected, append None
                processed_frames.append(None)
                face_det_flags.append(False)

        # select a single segment of >=5s of continuous frames with detected face from the video
        selected_frame_indices = self.select_frames(face_det_flags)
        if selected_frame_indices is None:
            # if such a segment is not available, stop here and return None
            return                                                                              # skip this video
        if config.VERBOSE:
            print(f"{os.path.basename(video_path)}: selected frames {selected_frame_indices[0]} to {selected_frame_indices[-1]} to split into clips")

        # initialise array to store ST maps for each clip
        num_clips = (len(selected_frame_indices) - config.CLIP_SIZE) // config.STRIDE + 1       # number of clips possible from selected segment of the video
        st_maps = np.zeros((num_clips, config.CLIP_SIZE, config.ROI_GRID_SIZE**2, 3))           # shape: (num_clips, T, n, c) where T = clip size, n = number of ROIs, c = three colour channels

        # generate ST maps for each clip
        for clip_idx, start_frame_idx_in_clip in enumerate(range(0, len(selected_frame_indices) - config.CLIP_SIZE + 1, config.STRIDE)):
            # select frames for the current clip
            final_frame_idx_in_clip = start_frame_idx_in_clip + config.CLIP_SIZE

            # for each frame in clip, compute average pixel value per ROI cell across all three colour channels
            st_map = np.zeros_like(st_maps[clip_idx, :, :, :])
            for frame_idx_in_clip, frame_idx_in_video in enumerate(selected_frame_indices[start_frame_idx_in_clip:final_frame_idx_in_clip]):
                roi_cells = self.get_roi_cells(processed_frames[frame_idx_in_video])
                for cell_idx, cell in enumerate(roi_cells):
                    avg_pixels = cv2.mean(cell)
                    st_map[frame_idx_in_clip, cell_idx, 0] = avg_pixels[0]
                    st_map[frame_idx_in_clip, cell_idx, 1] = avg_pixels[1]
                    st_map[frame_idx_in_clip, cell_idx, 2] = avg_pixels[2]

            # for each ROI cell, apply min-max norm. to each colour channel across all frames and scale to [0, 255]
            scaling_fn = lambda x: (x * 255.0).astype(np.uint8)
            for cell_idx in range(st_map.shape[1]):
                scaled_ch0 = self.min_max_norm.fit_transform(st_map[:, cell_idx, 0].reshape(-1, 1))
                st_map[:, cell_idx, 0] = scaling_fn(scaled_ch0.flatten())
                scaled_ch1 = self.min_max_norm.fit_transform(st_map[:, cell_idx, 1].reshape(-1, 1))
                st_map[:, cell_idx, 1] = scaling_fn(scaled_ch1.flatten())
                scaled_ch2 = self.min_max_norm.fit_transform(st_map[:, cell_idx, 2].reshape(-1, 1))
                st_map[:, cell_idx, 2] = scaling_fn(scaled_ch2.flatten())

            st_maps[clip_idx, :, :, :] = st_map

        if config.VERBOSE:
            print(f"{os.path.basename(video_path)}: ST maps generated for {num_clips} clips")
        return st_maps