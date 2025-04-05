import torch.nn as nn

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
            out_channels *= 2                                   # output channels are doubled after each block
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