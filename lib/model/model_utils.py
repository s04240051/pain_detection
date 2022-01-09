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