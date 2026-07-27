from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
from math import sqrt

class highwayNet(nn.Module):

    ## Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        #self-multi-parameters
        self.dim_in = args['dim_in']
        self.dim_k = args['dim_k']
        self.dim_v = args['dim_v']
        self.num_heads = args['num_heads']
        self.norm_fact = 1 / sqrt(self.dim_k//self.num_heads)

        self.input_embedding_size = args['input_embedding_size']

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # self
        self.position_emb = nn.Parameter(torch.rand((128, 3, 13)))
        self.graph_fl = torch.nn.Linear(39, 16)
        self.graph_fl2 = torch.nn.Linear(1, 32)

        # Encoder LSTM
        self.enc_lstm1 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Encoder LSTM
        self.enc_lstm2 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        self.spatial_embedding = nn.Linear(5, self.encoder_size)

        self.tanh = nn.Tanh()

        self.pre4att = nn.Sequential(
            nn.Linear(self.encoder_size, 1),
        )

        self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 2)  # 2-dimension (x, y)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        #self-multi-function-define
        self.linear_q = nn.Linear(self.dim_in, self.dim_k, bias = False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias = False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias = False)
        self.pre2att = nn.Sequential(nn.Linear(16, 1))

    def attention(self, lstm_out_weight, lstm_out):
        alpha = F.softmax(lstm_out_weight, 1)

        lstm_out = lstm_out.permute(0, 2, 1)

        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)
        new_hidden_state = F.relu(new_hidden_state)

        return new_hidden_state, alpha

    ## Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc, lon_enc, graph):
        # print(graph)

        if len(graph.shape) > 3:
            print(graph.shape)
            graph = graph[:, :, :, 0]
        graph = graph + self.position_emb
        graph = graph.unsqueeze(-1)
        graph = graph.permute(0, 3, 2, 1)
        graph = graph.flatten(2)
        # graph = graph.contiguous().view(graph.shape[0], graph.shape[1], -1)
        # graph = torch.tensor(graph.clone().detach, dtype=torch.float)
        graph_tem = self.leaky_relu(self.graph_fl(graph))
        graph_tem = graph_tem.permute(0, 2, 1)
        graph_tem = self.leaky_relu(self.graph_fl2(graph_tem))

        graph = graph_tem.permute(1, 0, 2)

        hist_tem = self.leaky_relu(self.ip_emb(hist))

        res = graph + hist_tem

        ## Forward pass hist:
        lstm_out, (hist_enc, _) = self.enc_lstm1(res)

        lstm_out = lstm_out.permute(1, 0, 2)

        batch, n, dim_in = lstm_out.shape
        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh
        q = self.linear_q(lstm_out).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(lstm_out).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(lstm_out).reshape(batch, n, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self.norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  #128， 16， 64
        new_hidden = att.permute(0, 2, 1)
        new_hidden = self.pre2att(new_hidden)


        nbrs_out, (nbrs_enc, _) = self.enc_lstm1(self.leaky_relu(self.ip_emb(nbrs)))
        # apply attention mechanism to neighbors
        nbrs_out = nbrs_out.permute(1, 0, 2)

        nbrs_lstm_weight = self.pre4att(self.tanh(nbrs_out))

        new_nbrs_hidden, soft_nbrs_attn_weights = self.attention(nbrs_lstm_weight, nbrs_out)
        nbrs_enc = new_nbrs_hidden

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()  # mask size: (128, 3, 13, 64)
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)

        masks_tem = masks.permute(0, 3, 2, 1)

        soc_enc = soc_enc.permute(0, 3, 2, 1)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)

        # concatenate hidden states:
        new_hs = torch.cat((soc_enc, new_hidden), 2)
        new_hs_per = new_hs.permute(0, 2, 1)

        # second attention
        weight = self.pre4att(self.tanh(new_hs_per))

        new_hidden_ha, soft_attn_weights_ha = self.attention(weight, new_hs_per)

        ## Concatenate encodings:
        enc = new_hidden_ha

        fut_pred = self.decode(enc)
        return fut_pred, soft_nbrs_attn_weights, soft_attn_weights_ha


    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)

        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred


