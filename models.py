import math

import torch
from torch import nn
from torch.nn import functional as F


def get_positional_value(pos: torch.int32, i: torch.int32, d_model: int) -> torch.Tensor:
    # pos是位置，i是维度下标
    f = torch.sin if i % 2 == 0 else torch.cos
    return f(pos / 1e4 ** (2 * i / d_model))


def get_positional_embedding(seq_length: int, d_model: int) -> torch.Tensor:
    positional_embedding = torch.zeros(seq_length, d_model)
    for pos in torch.arange(seq_length):
        for i in torch.arange(d_model):
            positional_embedding[pos, i] = get_positional_value(pos, i, d_model)
    return positional_embedding


class Attention(nn.Module):

    def __init__(self, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.drop_out = nn.Dropout(0.1)

    def forward(self, Q, K, V):
        # Q: [batch_size, seq_length, d_model]
        # K: [batch_size, seq_length, d_model]
        # V: [batch_size, seq_length, d_model]

        # 1. 计算注意力分数
        # Q @ K.transpose(-2, -1) 会得到 [batch_size, seq_length, seq_length]
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_model)

        # 2. Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 3. 与V相乘得到输出
        # [batch_size, seq_length, seq_length] @ [batch_size, seq_length, d_model]
        output = attention_weights @ V

        # 4. 添加Dropout
        output = self.drop_out(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *args, **kwargs):
        # TODO: 应该是有两层。
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.diff = 4 * d_model  # ff层的隐藏层是d_model的四倍
        self.proj1 = nn.Linear(self.d_model, self.diff)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(self.diff, self.d_model)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, input):
        proj1_res = self.proj1(input)
        proj2_input = self.drop_out(self.act(proj1_res))
        output = self.proj2(proj2_input)
        return output


class Decoder(nn.Module):

    def __init__(self, seq_length: int, vocab_size: int, embedding_dim: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # k, q, v projection
        self.k_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.q_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.v_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.attention = Attention(self.d_model)
        self.ff = FeedForward(self.d_model)
        self.batch_norm = nn.BatchNorm1d(self.d_model)

    def proj_embedding(self, embedding_input):
        return self.k_proj(embedding_input), self.q_proj(embedding_input), self.v_proj(embedding_input)

    def feed_forward(self, input):
        ff_res = self.ff(input)

        return self.batch_norm(input + ff_res)

    def forward(self, input_seq: torch.Tensor):
        # TODO: pad
        token_embedding_matricx = self.token_embedding(input_seq)
        pos_embedding = get_positional_embedding(self.seq_length, self.d_model)
        print(token_embedding_matricx.shape)
        print(pos_embedding.shape)  # OK, now the matrix shape is matched

        input_embedding = (token_embedding_matricx + pos_embedding) * math.sqrt(
            self.embedding_dim)  # times this to control the value range
        k, q, v = self.proj_embedding(input_embedding)  # kqv has the same size (sqe_length, hidden_dim)
        attention_res = self.attention(k, q, v)

        # residual_res = self.


class AttentionBlock(nn.Module):

    def __init__(self, d_model: int, embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.k_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.q_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.v_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.attention = Attention(self.d_model)
        self.ff = FeedForward(self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def proj_embedding(self, embedding_input):
        return self.k_proj(embedding_input), self.q_proj(embedding_input), self.v_proj(embedding_input)

    def feed_forward(self, input):
        ff_res = self.ff(input)

        return self.batch_norm(input + ff_res)

    def forward(self, input_embedding):
        k, q, v = self.proj_embedding(input_embedding)  # kqv has the same size (sqe_length, hidden_dim)
        attention_res = self.attention(k, q, v)
        residual_res = self.layer_norm(input_embedding + attention_res)
        ff_res = self.ff(residual_res)
        output = self.layer_norm(residual_res + ff_res)
        return output


class Encoder(nn.Module):

    def __init__(self, seq_length: int, vocab_size: int, embedding_dim: int, d_model: int, n_layer: int, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attention_blocks = nn.ModuleList(
            AttentionBlock(d_model, embedding_dim) for _ in range(n_layer)
        )

    def forward(self, input_seq: torch.Tensor):
        # TODO: pad
        token_embedding_matricx = self.token_embedding(input_seq)
        pos_embedding = get_positional_embedding(self.seq_length, self.d_model)
        input_embedding = (token_embedding_matricx + pos_embedding) * math.sqrt(
            self.embedding_dim)  # times this to control the value range
        output = None
        for layer in self.attention_blocks:
            output = layer(input_embedding)

        return output


if __name__ == "__main__":
    d_model = embedding_dim = 128
    # d_model represents the size of the hidden dimension,
    # embedding_dim represent the size of the the encoded tensor
    vocab_size = 1000
    seq_length = 32
    n_layer = 6
    input_sqe = torch.arange(seq_length)
    encoder = Encoder(seq_length=seq_length, vocab_size=vocab_size, embedding_dim=embedding_dim, d_model=d_model,
                      n_layer=n_layer)
    encoded_tensor = encoder(input_sqe)
    print(encoded_tensor.shape)

    pass
