import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels= 4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding= kernel_size // 2)
    
    def forward(self, x, hidden):

        h_prev, c_prev = hidden

        c_out = self.conv(torch.cat([x, h_prev], dim=1))

        i, f, g, o = torch.split(c_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim:list, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()

        self.layers.append(ConvLSTMCell(input_dim, hidden_dim[0], kernel_size))

        for i in range(1, num_layers):
            self.layers.append(ConvLSTMCell(hidden_dim[i - 1], hidden_dim[i], kernel_size))


    def forward(self, x, hidden=None):
        batch_size, seq_len, _, height, width = x.size()
        if hidden is None:
            hidden = [(torch.zeros(batch_size, h_dim, height, width, device=x.device),
                       torch.zeros(batch_size, h_dim, height, width, device=x.device)) 
                      for h_dim in self.hidden_dim]

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t]
            for layer_idx in range(self.num_layers):
                hidden[layer_idx] = self.layers[layer_idx](input_t, hidden[layer_idx])
                input_t = hidden[layer_idx][0]
            outputs.append(input_t)

        return torch.stack(outputs, dim=1)
