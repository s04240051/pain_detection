import torch.nn as nn
import torch
from .conv_LSTM import ConvLSTM
from . import model_utils as utils
import torch.nn.functional as F


class Two_stream_model(nn.Module):
    def __init__(self, cfg):

        super(Two_stream_model, self).__init__()
        self.clstm = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, 
            cfg.MODEL.NUM_CLSTM_LAYERS, 
            cfg.MODEL.IMG_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        self.lstm_stream = Kp_lstm(
            input_size= cfg.MODEL.LSTM_INPUT_SIZE, 
            num_layers= cfg.MODEL.NUM_LSTM_LAYERS, 
            hidden_size= cfg.MODEL.LSTM_HIDDEN_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        fuse_dim = self.clstm.linear_input + cfg.MODEL.LSTM_HIDDEN_SIZE
        self.head = utils.Head(cfg, fuse_dim)
        

    def forward(self, x):
        frame = x[0]
        kp = x[1]
        frame = self.clstm(frame)
        kp = self.lstm_stream(kp)
        f_k_fuse = torch.concat([frame, kp], 1)
        final_out = self.head(f_k_fuse)
        return final_out

class Two_stream_fusion(nn.Module):
    def __init__(self, cfg):
        super(Two_stream_fusion, self).__init__()
        self.clstm = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, 
            cfg.MODEL.NUM_CLSTM_LAYERS, 
            cfg.MODEL.IMG_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        self.lstm_stream = Kp_lstm(
            input_size= cfg.MODEL.LSTM_INPUT_SIZE, 
            num_layers= cfg.MODEL.NUM_LSTM_LAYERS, 
            hidden_size= cfg.MODEL.LSTM_HIDDEN_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        image_dim = self.clstm.linear_input
        kp_dim = cfg.MODEL.LSTM_HIDDEN_SIZE
        self.fusion_method = eval(f"utils.{cfg.MODEL.FUSION_METHOD}(image_dim, kp_dim, cfg)")
    
    def forward(self, x):
        frame = x[0]
        kp = x[1]
        frame = self.clstm(frame)
        kp = self.lstm_stream(kp)
        final_out = self.fusion_method(frame, kp)
        return final_out

class Flow_model(nn.Module):
    def __init__(self, cfg) :
        super(Flow_model, self).__init__()
        self.rgb_stream = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, 
            cfg.MODEL.NUM_CLSTM_LAYERS, 
            cfg.MODEL.IMG_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        self.flow_stream = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, 
            cfg.MODEL.NUM_CLSTM_LAYERS, 
            cfg.MODEL.IMG_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        
        linear_input = self.rgb_stream.linear_input
        self.fusion_method = eval(f"utils.{cfg.MODEL.FUSION_METHOD}(linear_input, linear_input, cfg)")
    
    def forward(self, x):
        rgb = x[0]
        flow = x[1]
        rgb_vector = self.rgb_stream(rgb)
        flow_vector = self.flow_stream(flow)
        final_out = self.fusion_method(rgb_vector, flow_vector)
        return final_out

class Rgb_model(nn.Module):
    def __init__(self, cfg):
        super(Rgb_model, self).__init__()
        self.clstm = Img_stream(
            cfg.MODEL.CLSTM_HIDDEN_SIZE, 
            cfg.MODEL.NUM_CLSTM_LAYERS, 
            cfg.MODEL.IMG_SIZE,
            with_attention=cfg.MODEL.ATTENTION,
        )
        out_units = 1 if cfg.MODEL.NUM_LABELS == 2 else cfg.MODEL.NUM_LABELS
        self.fc = nn.Linear(self.clstm.linear_input, out_units)
    def forward(self, x):
        frame = x[0]
        frame = self.clstm(frame)
        out = self.fc(frame)
        return out

class Img_stream(nn.Module):
    def __init__(
        self,
        nb_units,
        nb_layers,
        input_size,
        nb_labels=2,
        with_attention=True,
        top_layer=False,
    ):

        super(Img_stream, self).__init__()
        self.top_layer = top_layer
        self.with_attention = with_attention
        self.convlstm = ConvLSTM(
            3, nb_units, (5, 5), nb_layers, True, True, False)
        input_width, input_height = input_size
        self.linear_input = (input_height//(2**nb_layers)) * \
            (input_width//(2**nb_layers))*nb_units
        out_units = 1 if nb_labels == 2 else nb_labels
        self.fc = nn.Linear(self.linear_input, out_units)
        self.timedistribute = utils.TimeDistributed(nn.Flatten(1))
        self.flat = nn.Flatten(1)
        self.attention = utils.TemporalAttn(self.linear_input)

    def forward(self, x):
        output, h = self.convlstm(x)  # (batch, time, hidden_size, w, h)
        if self.with_attention:
            x = output[0]
            x = self.timedistribute(x)
            x, weight = self.attention(x)  # (batch, hidden_size)
        else:
            x = h[0][0]
            x = self.flat(x)
        
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
        with_attention=True,
        top_layer=False,
    ):

        super(Kp_lstm, self).__init__()
        self.with_attention = with_attention
        self.top_layer = top_layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = utils.TemporalAttn(hidden_size)
        self.fc = nn.Linear(hidden_size, nb_labels)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        if self.with_attention:
            x, weights = self.attention(x)
        else:
            x = h_n.permute(1,0,2)[:,-1,:]
        if self.top_layer:
            out = self.fc(x)
            return out
        else:
            return x


if __name__ == "__main__":
    
    import sys
    sys.path.append("..")
    
    from config_file import cfg

    
    model = Two_stream_model(cfg)
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