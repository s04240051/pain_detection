from turtle import forward
from block import fusions
import torch.nn.functional as F
import torch.nn as nn
import torch

class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        input_para = [-1] + [input_seq.size(item) for item in range(2, input_seq.dim())]
     
        reshaped_input = input_seq.contiguous().view(*input_para)
        output = self.module(reshaped_input)
        
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            out_para = [input_seq.size(0), -1] + [output.size(item) for item in range(1, output.dim())]
          
            output = output.contiguous().view(*out_para)
            
        else:
            # (timesteps, samples, output_size)
            out_para = [-1, input_seq.size(1)] + [output.size(item) for item in range(1, output.dim())]
            
            output = output.contiguous().view(*out_para)
        return output

class Head(nn.Module):
    def __init__(self, cfg, vector_dim):
        super(Head, self).__init__()
        out_units = 1 if cfg.DATA.DATA_TYPE in ["simple","aux" ] else 3
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1), 
            nn.Linear(vector_dim, out_units),
        )
        if cfg.DATA.REQUIRE_AUX:
            self.aux_fc = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(vector_dim, 3),
            )
            self.two_out = True
        else:
            self.two_out = False

    def forward(self, x):
        out = self.fc(x)
        if self.two_out:
            out2 = self.aux_fc(x)
            return [out, out2]
        else:
            return out
class Late(nn.Module):
    def __init__(self, dim1, dim2, cfg):
        super(Late, self).__init__()
        self.head1 = Head(cfg, dim1)
        self.head2 = Head(cfg, dim2)
        self.act = nn.Sigmoid() if cfg.DATA_TYPE == "simple" else nn.Softmax()
        weight = torch.rand(2, requires_grad=True)
        self.weight = torch.nn.Parameter(weight)
    def forward(self,x1, x2):
        x1 = self.head1(x1)
        x2 = self.head2(x2)
        score1 = self.act(x1)
        score2 = self.act(x2)
        weight_sum = torch.sum(self.weight)
        out = (self.weight[0]/weight_sum)*score1 + (self.weight[1]/weight_sum)*score2
        return out

class Bilinear(nn.Module):
    def __init__(self, dim1, dim2, cfg):
        super(Bilinear, self).__init__()
        dim_out = cfg.MODEL.BILINEAR_OUT_DIM
        self.bilinear_fusion = fusions.Mutan([dim1, dim2], dim_out)
    def forward(self, x1, x2):
        out = self.bilinear_fusion([x1, x2])
        return out
class Concat(nn.Module):
    def __init__(self, *args):
        super(Concat, self).__init__()
        
    def forward(self, x1, x2):
        out = torch.concat([x1, x2], 1)
        return out

