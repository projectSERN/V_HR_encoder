import torch.nn as nn
import torch

from .config import config

class DFD_Conv1d(nn.Module):
    """ Temporal convolution model for deepfake detection using V_HR modality.

    Attibutes:
        conv_model (nn.Sequential): Temporal convolution model with multiple Conv1d, ReLU and MaxPool1d layers.
        dropout (nn.Dropout): Dropout layer after the temporal convolution model.
        fc_layer (nn.Sequential): Fully connected layer to get the feature representation.
        classifier (nn.Sequential): Classifier head to get the deepfake detection prediction.
    """
    def __init__(self):
        super(DFD_Conv1d, self).__init__()

        # temporal convolution model with config.NUM_BLOCKS num. of conv blocks
        layers = []
        in_channels = 1                                         # since input is of shape (batch_size, 1, num_clips)
        out_channels = 16

        for i in range(config.NUM_BLOCKS):
            # append Conv1d, BatchNorm1d, ReLU and MaxPool1d layers for current block
            layers.append(nn.Conv1d(in_channels, out_channels, config.KERNEL_SIZE))   # padding, stride, dilation are set to default values
            if i % 2 == 1:                                      # add batch norm layer before ReLU in odd blocks
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(config.POOLING_SIZE))
            # update in_channels and out_channels for next block
            in_channels = out_channels
            out_channels *= 2                # MORE GRADUAL INCREASE??                # output channels are doubled after each block
        self.conv_model = nn.Sequential(*layers)

        # dropout layer
        self.dropout = None
        if config.DROPOUT_RATE > 0.0:
            self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # FC layer
        conv_output_dim = (2**(config.NUM_BLOCKS - 1) * 16)     # dim 1 of the input tensor after passing through the conv model
        seq_len = self.compute_seq_len()                        # dim 2 of the input tensor after passing through the conv model
        self.fc_layer = nn.Sequential(
            nn.Linear(conv_output_dim * seq_len, 256),          # to get feature representation of shape (batch_size, 256)
            nn.ReLU()
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.init_weights()                                     # initialise weights of the model

    def compute_seq_len(self):
        """ Compute the sequence length of the input tensor after passing through the conv model. """
        seq_len = (300 - config.CLIP_SIZE) // config.STRIDE + 1                         # num_clips
        for _ in range(config.NUM_BLOCKS):
            # After Conv1d
            seq_len = (seq_len - (config.KERNEL_SIZE - 1) - 1) + 1                      # accounted for default padding, stride and dilation
            # After MaxPool1d
            seq_len = (seq_len - config.POOLING_SIZE) // config.POOLING_SIZE + 1        # accounted for default padding, stride and dilation
        return seq_len

    def init_weights(self):
        """ Initialise weights of the temporal convolution model. """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Forward pass of the temporal convolution model.

        Args:
            x (torch.Tensor): Input tensor with V_HR sequences of shape (batch_size, 1, num_clips).

        Returns:
            preds (torch.Tensor): Predictions from the model (i.e., after the classifier head).
            feats (torch.Tensor): Feature representations from the model (i.e., before the classifier head).
        """
        feats = self.conv_model(x)

        if self.dropout is not None:
            feats = self.dropout(feats)

        feats = self.fc_layer(feats.view(feats.size(0), -1))    # flatten dims 1 and 2 before passing to the FC layer
        preds = self.classifier(feats).squeeze(-1)
        return preds, feats                                     # preds shape: (batch_size,), feats shape: (batch_size, 256)

class DFD_LSTM(nn.Module):
    """ LSTM model for deepfake detection using V_HR modality.

    Attributes:
        layers (int): Number of LSTM layers.
        layer{n} (nn.LSTM): n-th LSTM layer.
        dropout (nn.Dropout): Dropout layer after the LSTM model.
        fc_layer (nn.Sequential): Fully connected layer to get the feature representation.
        classifier (nn.Sequential): Classifier head to get the deepfake detection prediction.
    """
    def __init__(self):
        super(DFD_LSTM, self).__init__()

        # LSTM layers
        self.layers = config.NUM_LAYERS
        input_dim = 1
        for layer in range(config.NUM_LAYERS):
            setattr(self, f'layer{layer}', nn.LSTM(input_dim, config.HIDDEN_SIZE, batch_first=True, dropout=config.DROPOUT_RATE))
            input_dim = config.HIDDEN_SIZE

        # dropout layer
        self.dropout = None
        if config.DROPOUT_RATE > 0.0:
            self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # FC layer
        self.fc_layer = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, 256),
            nn.ReLU()
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.init_weights()                                     # initialise weights of the model

    def init_weights(self):
        """ Initialise weights of the LSTM model. """
        for module in self.modules():
            if type(module) in [nn.LSTM, nn.RNN, nn.GRU]:
                # _hh_: hidden-hidden, _ih_: input-hidden
                nn.init.orthogonal_(module.weight_hh_l0)
                nn.init.xavier_uniform_(module.weight_ih_l0)
                nn.init.zeros_(module.bias_hh_l0)
                nn.init.zeros_(module.bias_ih_l0)
            elif type(module) in [nn.Linear]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """ Forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor with heart rate sequences of shape (batch_size, num_clips, 1).

        Returns:
            preds (torch.Tensor): Predictions from the model (i.e., after the classifier head).
            feats (torch.Tensor): Feature representations from the model (i.e., before the classifier head).
        """
        x = x.permute(0, 2, 1)                                  # permute to (batch_size, num_clips, 1), required for LSTM input

        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht[-1]                                          # shape: (batch_size, config.HIDDEN_SIZE)

        if self.dropout is not None:
            feats = self.dropout(feats)

        feats = self.fc_layer(feats)                            # no need to flatten before passing to the FC layer
        preds = self.classifier(feats).squeeze(-1)
        return preds, feats                                     # preds shape: (batch_size,), feats shape: (batch_size, 256)

class DFD_Ensemble(nn.Module):
    """ Ensemble model for deepfake detection using V_HR modality, combining Conv1d and LSTM architectures.

    Attributes:
        conv_model (DFD_Conv1d): The temporal convolution model component.
        lstm_model (DFD_LSTM): The LSTM model component.
        fusion_method (str): Method to combine model outputs ('concat', 'average', or 'weighted').
        fusion_weights (tuple): Weights for weighted fusion method (conv_weight, lstm_weight).
        fusion_layer (nn.Sequential): Fusion layer if using concatenation fusion method.
        classifier (nn.Sequential): Classifier head for the ensemble model.
    """
    def __init__(self, fusion_method='concat', fusion_weights=(0.5, 0.5)):
        super(DFD_Ensemble, self).__init__()

        # Initialize both base models
        self.conv_model = DFD_Conv1d()
        self.lstm_model = DFD_LSTM()

        # Fusion method and parameters
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights

        # Fusion layer (for concatenation method)
        if fusion_method == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(256 + 256, 256),  # Combine features from both models (256 from each)
                nn.ReLU()
            )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """ Initialize weights of the fusion layer and classifier. """
        if self.fusion_method == 'concat':
            for m in self.fusion_layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Forward pass of the ensemble model.

        Args:
            x (torch.Tensor): Input tensor with V_HR sequences.
                For Conv1d: shape (batch_size, 1, num_clips)
                For LSTM: Will be permuted to (batch_size, num_clips, 1)

        Returns:
            preds (torch.Tensor): Final predictions from the ensemble model.
            feats (torch.Tensor): Final feature representations from the ensemble model.
        """
        # Process through Conv1d model
        conv_preds, conv_feats = self.conv_model(x)

        # Process through LSTM model
        lstm_preds, lstm_feats = self.lstm_model(x)

        # Fusion based on specified method
        if self.fusion_method == 'concat':
            # Concatenate features and pass through fusion layer
            combined_feats = torch.cat((conv_feats, lstm_feats), dim=1)
            feats = self.fusion_layer(combined_feats)
            preds = self.classifier(feats).squeeze(-1)

        elif self.fusion_method == 'average':
            # Average predictions
            preds = (conv_preds + lstm_preds) / 2
            # Average features (for consistency in return)
            feats = (conv_feats + lstm_feats) / 2

        elif self.fusion_method == 'weighted':
            # Weighted average of predictions
            conv_weight, lstm_weight = self.fusion_weights
            preds = conv_weight * conv_preds + lstm_weight * lstm_preds
            # Weighted average of features (for consistency in return)
            feats = conv_weight * conv_feats + lstm_weight * lstm_feats

        else:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")

        return preds, feats

class DFD_Conv1d_LSTM(nn.Module):
    """ Sequential Conv1d-LSTM model for deepfake detection using V_HR modality.

    This model first processes the heart rate sequence with temporal convolutions
    to extract local patterns, then feeds these features into an LSTM to capture
    long-range dependencies.

    Attributes:
        conv_layers (nn.Sequential): Temporal convolution layers.
        lstm_layers (nn.ModuleList): LSTM layers.
        dropout (nn.Dropout): Dropout layer.
        fc_layer (nn.Sequential): Fully connected layer for feature representation.
        classifier (nn.Sequential): Classifier head for prediction.
    """
    def __init__(self):
        super(DFD_Conv1d_LSTM, self).__init__()

        # Temporal convolution layers
        layers = []
        in_channels = 1                                # Input has shape (batch_size, 1, num_clips)
        out_channels = 16

        for i in range(config.NUM_BLOCKS):
            # Conv1d, optional BatchNorm, ReLU, and MaxPool1d for each block
            layers.append(nn.Conv1d(in_channels, out_channels, config.KERNEL_SIZE, padding=config.KERNEL_SIZE//2))
            if i % 2 == 1:  # Add batch norm in odd blocks
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(config.POOLING_SIZE))

            # Update channels for next block
            in_channels = out_channels
            out_channels *= 2  # Double channels after each block

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output shape after conv layers
        # This is needed to properly connect to the LSTM
        num_clips = (300 - config.CLIP_SIZE) // config.STRIDE + 1  # Initial sequence length

        # Calculate how the sequence length changes through conv and pooling layers
        for _ in range(config.NUM_BLOCKS):
            # After Conv1d with padding=kernel_size//2, sequence length is unchanged
            # After MaxPool1d
            num_clips = (num_clips - config.POOLING_SIZE) // config.POOLING_SIZE + 1

        # Calculate conv output features
        conv_out_channels = 16 * (2 ** (config.NUM_BLOCKS - 1))

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = conv_out_channels  # Input to first LSTM is output from conv

        for _ in range(config.NUM_LAYERS):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=config.HIDDEN_SIZE,
                    batch_first=True,
                    dropout=config.DROPOUT_RATE if config.NUM_LAYERS > 1 else 0
                )
            )
            lstm_input_size = config.HIDDEN_SIZE  # Subsequent LSTM layers take previous hidden size

        # Dropout layer
        self.dropout = None
        if config.DROPOUT_RATE > 0.0:
            self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # FC layer for feature representation
        self.fc_layer = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, 256),
            nn.ReLU()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        """ Initialize weights of the model. """
        # Initialize Conv1d layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == 1:  # Sigmoid activation for classifier
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                else:  # ReLU activation for other linear layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        # Initialize LSTM layers
        for lstm in self.lstm_layers:
            # Initialize weights using orthogonal initialization
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        """ Forward pass of the Conv1d-LSTM sequential model.

        Args:
            x (torch.Tensor): Input tensor with V_HR sequences of shape (batch_size, 1, num_clips).

        Returns:
            preds (torch.Tensor): Predictions from the model.
            feats (torch.Tensor): Feature representations from the model.
        """
        # Process through Conv1d layers
        x = self.conv_layers(x)  # Shape: (batch_size, conv_out_channels, seq_len_after_conv)

        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len_after_conv, conv_out_channels)

        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x, (h_n, _) = lstm(x)

        # Use the final hidden state from the last LSTM layer
        feats = h_n[-1]  # Shape: (batch_size, hidden_size)

        # Apply dropout if specified
        if self.dropout is not None:
            feats = self.dropout(feats)

        # Process through FC layer
        feats = self.fc_layer(feats)  # Shape: (batch_size, 256)

        # Final classification
        preds = self.classifier(feats).squeeze(-1)  # Shape: (batch_size,)

        return preds, feats  # preds shape: (batch_size,), feats shape: (batch_size, 256)