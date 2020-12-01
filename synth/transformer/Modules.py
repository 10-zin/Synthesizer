import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# template for attention synthesizers
class DenseAttention(nn.Module):
    def __init__(self, max_seq_len, d_k, d_hid = 64, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid)
        self.w_2 = nn.Linear(d_hid, max_seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None):

        # b x n x lq x dq -> b x n x lq x lq #
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:,:,:,:len_q]
        # print('Attn: ', dense_attn.shape)
        # print('Mask: ', mask.shape)
        # print('V: ', v.shape)

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn

class FactorizedDenseAttention(nn.Module):
    def __init__(self, max_seq_len, d_k, f, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(DenseAttention, self).__init__()
        self.f = f
        self.max_seq_len = max_seq_len
        self.f_a = nn.Linear(d_k, f)
        self.f_b = nn.Linear(d_k, max_seq_len/f)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None, factorize=False):

        h_a = torch.repeat_interleave(self.f_a(q), self.max_seq_len/self.f, -1)[:,:,:,:len_q]
        h_b = torch.repeat_interleave(self.f_b(q), self.f, -1)[:,:,:,:len_q]
        dense_attn = torch.matmul(h_a, h_b.transpose(2, 3))

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn

class RandomAttention(nn.Module):
    def __init__(self, batch_size, n_head, max_seq_len, attn_dropout = 0.1):
        super(RandomAttention, self).__init__()
        #device = torch.device("GPU"),
        self.random_attn = torch.randn(batch_size, n_head, max_seq_len, max_seq_len, requires_grad = True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, v, len_q, mask=None):

        # b x n x max_len x max_len -> b x n x lq x lq
        random_attn = self.random_attn[:mask.shape[0],:,:len_q,:len_q]
        random_attn = random_attn.to(torch.device('cuda' if mask.is_cuda else 'cpu'))

        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)

        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)
        
        return output, random_attn

class FactorizedRandomAttention(nn.Module):
    def __init__(self, batch_size, n_head, f,  max_seq_len, attn_dropout = 0.1):
        super(RandomAttention, self).__init__()
        #device = torch.device("GPU"),
        self.random_attn_1 = torch.randn(batch_size, n_head, max_seq_len, f, requires_grad = True)
        self.random_attn_2 = torch.randn(batch_size, n_head, f, max_seq_len, requires_grad = True)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, v, len_q, mask=None, factorize=False):
        # b x n x max_len x max_len -> b x n x lq x lq #[:,:,:len_q,:len_q]
        random_attn = torch.matmul(self.random_attn_1, self.random_attn_2)[:mask.shape[0],:,:len_q,:len_q]

        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)

        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)
        
        return output, random_attn