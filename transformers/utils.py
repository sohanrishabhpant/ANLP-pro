import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        #print(x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNormalisation(nn.Module):
    def __init__(self, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        print(scores.shape)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return attn @ value, attn
    
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q).view(q.size(0), q.size(1), self.n_head, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(k.size(0), k.size(1), self.n_head, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(v.size(0), v.size(1), self.n_head, self.d_k).transpose(1, 2)
        x, attn = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_head * self.d_k)
        return self.w_o(x)

class ResidualConnections(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm = LayerNormalisation()
        self.dropout = nn.Dropout(dropout) 
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))