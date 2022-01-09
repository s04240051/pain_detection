from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
from .conv_LSTM import ConvLSTM
from .model_utils import TemporalAttn, TimeDistributed
import torch.nn.functional as F


class two_stream_model(nn.Module):
    def __init__(self, cfg):

        super(two_stream_model, self).__init__()
        self.clstm = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, cfg.MODEL.NUM_CLSTM_LAYERS, cfg.MODEL.IMG_SIZE
        )
        self.lstm_stream = Kp_lstm(
            cfg.MODEL.LSTM_INPUT_SIZE, cfg.MODEL.NUM_LSTM_LAYERS, cfg.MODEL.LSTM_HIDDEN_SIZE
        )
        fuse_dim = self.clstm.linear_input + cfg.MODEL.LSTM_HIDDEN_SIZE
        out_units = 1 if cfg.MODEL.NUM_LABELS == 2 else cfg.MODEL.NUM_LABELS
        self.fc = nn.Linear(fuse_dim, out_units)
        

    def forward(self, x):
        frame = x[0]
        kp = x[1]
        frame = self.clstm(frame)
        kp = self.lstm_stream(kp)
        f_k_fuse = torch.concat([frame, kp], 1)
        out = self.fc(f_k_fuse)
        
        return out


class Img_stream(nn.Module):
    def __init__(
        self,
        nb_units,
        nb_layers,
        input_size,
        nb_labels=2,
        top_layer=False,
    ):

        super(Img_stream, self).__init__()
        self.top_layer = top_layer
        self.convlstm = ConvLSTM(
            3, nb_units, (5, 5), nb_layers, True, True, False)
        input_width, input_height = input_size
        self.linear_input = (input_height//(2**nb_layers)) * \
            (input_width//(2**nb_layers))*nb_units
        out_units = 1 if nb_labels == 2 else nb_labels
        self.fc = nn.Linear(self.linear_input, out_units)
        self.timedistribute = TimeDistributed(nn.Flatten(1))
        
        self.attention = TemporalAttn(self.linear_input)

    def forward(self, x):
        x = self.convlstm(x)[0][0]  # (batch, time, hidden_size, w, h)

        x = self.timedistribute(x)

        x, weight = self.attention(x)  # (batch, hidden_size)
        if self.top_layer:
            x = self.fc(x)
            
            return x
        else:
            return x


class Kp_lstm(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        hidden_size,
        nb_labels=2,
        top_layer=False,
    ):

        super(Kp_lstm, self).__init__()
        self.top_layer = top_layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = TemporalAttn(hidden_size)
        self.fc = nn.Linear(hidden_size, nb_labels)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x, weights = self.attention(x)
        if self.top_layer:
            out = self.fc(x)
            return out
        else:
            return x


if __name__ == "__main__":
    
    import sys
    sys.path.append("..")
    
    from config_file import cfg

    
    model = two_stream_model(cfg)
    model = model.cuda()
    img = torch.rand((1, 10, 3, 224, 224))
    kp = torch.rand((1, 10, 34))
    img = img.cuda()
    kp = kp.cuda()
    input = [img, kp]
    out = model(input)
    print(out.size())
    '''
    x = torch.rand((32, 10, 3, 128, 128))
    model = Img_stream(top_layer=False)
    out = model(x)
    print(out.size())  # (32, 10, 32, 8, 8)
    '''