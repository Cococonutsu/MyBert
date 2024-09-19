import torch
import torch.nn as nn


class Feed_Forward(nn.Module):
    def __init__(self, hidden_size, feed_num):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, feed_num)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(feed_num, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Add_Norm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.add = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.add(x)
        x = self.layer_norm(x)
        return x


class Multi_Head_Attention(nn.Module):
    def __init__(self, hidden_size, head_num):
        super().__init__()
        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

    # def forward(self, x):  # batch, seq_len, hidden_size
    #     batch, seq_len, hidden_size = x.shape  # batch, seq_len, hidden_size
    #     x = x.reshape(batch, self.head_num, -1, hidden_size)  # batch, head_num, (seq_len//head_num), hidden_size
    #     q = self.Q(x)  # batch, head_num, (seq_len//head_num), hidden_size
    #     k = torch.transpose(self.K(x), -1, -2)  # batch, head_num, (seq_len//head_num), hidden_size
    #     v = self.V(x)  # batch, head_num, (seq_len//head_num), hidden_size
    #     att = self.softmax((q @ k) / hidden_size) @ v
    #     att = att.reshape(batch, seq_len, hidden_size)
    #     return att#([40, 128, 768])

    def forward(self, x):  # batch, seq_len, hidden_size
        batch, seq_len, hidden_size = x.shape  # batch, seq_len, hidden_size
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.reshape(batch, seq_len, self.head_num, -1).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.head_num, -1).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.head_num, -1).transpose(1, 2)
        att = self.softmax((q @ k.transpose(-1, -2)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))) @ v
        att = att.transpose(1, 2).reshape(batch, seq_len, hidden_size)
        return att


class Bert_Encoder(nn.Module):
    def __init__(self, hidden_size, feed_num, head_num):
        super().__init__()
        self.multi_head_attention = Multi_Head_Attention(hidden_size, head_num)
        self.add_norm_1 = Add_Norm(hidden_size)
        self.feed_forward = Feed_Forward(hidden_size, feed_num)
        self.add_norm_2 = Add_Norm(hidden_size)

    def forward(self, x):
        atten_out = self.multi_head_attention(x)
        add_norm_1_out = self.add_norm_1(atten_out)
        add_norm_1_out += x
        feed_forward_out = self.feed_forward(add_norm_1_out)
        add_norm_2_out = self.add_norm_2(feed_forward_out)
        add_norm_2_out += add_norm_1_out
        return add_norm_2_out


class Cls_Attention(nn.Module):
    def __init__(self, hidden_size, head_num):
        super().__init__()
        self.head_num = head_num
        self.linear = nn.Linear(hidden_size // head_num, hidden_size // head_num)
        self.relu = nn.ReLU()

    def forward(self, x):  # batch_size hiddem_size
        batch_size, hidden_size = x.shape
        x = x.reshape(batch_size, self.head_num, -1)  # batch_size head_num (hiddem_size//head_num)
        x = self.linear(x)  # batch_size head_num (hiddem_size//head_num)
        x = self.relu(x)  # batch_size head_num (hiddem_size//head_num)
        x = x.reshape(batch_size, hidden_size)
        return x
