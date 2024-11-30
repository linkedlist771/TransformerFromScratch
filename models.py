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
        return output


class MaskedAttention(nn.Module):

    def __init__(self, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model

    def forward(self, Q, K: torch.Tensor, V):
        # batch_size x seq_length x d_model
        # 首先实现基本的attention 然后加上mask
        # attent(Q, K, V) = softmax(Q*K^T/sqrt{d_model}) * V
        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        # 无所谓哎
        # 然后加上mask
        mask = torch.ones_like(attention_scores)  # mask 应该在softmax 之前
        # 然后制作一个下三角矩阵，
        tri = torch.tril(mask)
        # 其他地方被置为0了， 但是实际上是为-inf
        attention_scores[tri == 0] = float('-inf')
        attention_scores = F.softmax(attention_scores, dim=-1)
        atten = attention_scores @ V
        return atten


class AttentionWithWeight(nn.Module):

    def __init__(self, d_model: int, head_hid: int, mask: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.head_hid = head_hid
        self.k_proj = nn.Linear(self.d_model, self.head_hid)
        self.q_proj = nn.Linear(self.d_model, self.head_hid)
        self.v_proj = nn.Linear(self.d_model, self.head_hid)
        self.atten = Attention(self.d_model) if not mask else MaskedAttention(self.d_model)

    def forward(self, Q, K, V):
        Q_prj = self.q_proj(Q)
        K_prj = self.k_proj(K)
        V_prj = self.v_proj(V)
        return self.atten(Q_prj, K_prj, V_prj)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, mask: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        # print(self.d_model % self.num_heads)
        assert not self.d_model % self.num_heads, "num of heads can not be divided by the d model"

        self.heads = nn.ModuleList(
            AttentionWithWeight(self.d_model, self.d_model // self.num_heads, mask) for _ in range(self.num_heads))
        # 这里还有一层投影的
        self.proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, Q, K, V):
        # return self.proj(torch.hstack([head(Q, K, V) for head in self.heads]))
        # fix: h stack has been replaced with the cat to make the batch compatiable
        return self.proj(torch.cat([head(Q, K, V) for head in self.heads], dim=-1))



class FeedForward(nn.Module):
    def __init__(self, d_model: int, *args, **kwargs):
        # TODO: 应该是有两层。
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.diff = 4 * d_model  # ff层的隐藏层是d_model的四倍
        self.proj1 = nn.Linear(self.d_model, self.diff)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(self.diff, self.d_model)

    def forward(self, input):
        proj1_res = self.proj1(input)
        proj2_input = self.act(proj1_res)
        output = self.proj2(proj2_input)
        return output


class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, embedding_dim: int, num_heads: int, mask: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.q_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.v_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.attention = MultiHeadAttention(self.d_model, self.num_heads, mask)
        self.ff = FeedForward(self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.drop_out = nn.Dropout(0.1)

    def proj_embedding(self, embedding_input):
        return self.k_proj(embedding_input), self.q_proj(embedding_input), self.v_proj(embedding_input)

    def feed_forward(self, input):
        ff_res = self.ff(input)

        return self.layer_norm(input + ff_res)

    def forward(self, input_embedding):
        k, q, v = self.proj_embedding(input_embedding)  # kqv has the same size (sqe_length, hidden_dim)
        attention_res = self.attention(k, q, v)
        attention_res = self.drop_out(attention_res)
        residual_res = self.layer_norm(input_embedding + attention_res)
        ff_res = self.ff(residual_res)
        ff_res = self.drop_out(ff_res)

        output = self.layer_norm(residual_res + ff_res)
        return output


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, embedding_dim: int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Projections for self-attention 投影输入的，
        self.self_attn_q_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.self_attn_k_proj = nn.Linear(self.embedding_dim, self.d_model)
        self.self_attn_v_proj = nn.Linear(self.embedding_dim, self.d_model)

        # Projections for cross-attention， 投影encoder的输入的
        self.cross_attn_q_proj = nn.Linear(self.d_model, self.d_model)
        self.cross_attn_k_proj = nn.Linear(self.d_model, self.d_model)
        self.cross_attn_v_proj = nn.Linear(self.d_model, self.d_model)

        # Masked multi-head self-attention
        self.mask_attention = MultiHeadAttention(self.d_model, self.num_heads, mask=True)
        # Cross-attention
        self.cross_attention = MultiHeadAttention(self.d_model, self.num_heads, mask=False)
        # Feed-forward network
        self.ff = FeedForward(self.d_model)

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.layer_norm3 = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, encoder_output):
        # x: [batch_size, tgt_seq_length, embedding_dim]
        # encoder_output: [batch_size, src_seq_length, d_model]

        # Masked Multi-Head Self-Attention
        q = self.self_attn_q_proj(x)
        k = self.self_attn_k_proj(x)
        v = self.self_attn_v_proj(x)
        masked_self_attn_output = self.mask_attention(q, k, v)
        masked_self_attn_output = self.dropout(masked_self_attn_output)
        x = self.layer_norm1(x + masked_self_attn_output)
        # 上面的部分是把input的embedding进行了mask multi head attention 的计算。

        # Multi-Head Cross-Attention
        q = self.cross_attn_q_proj(x)  # 然后把input 的embedding作为query 来在计算一个multihead的attention， 最后进行feedfoward.
        k = self.cross_attn_k_proj(encoder_output)
        v = self.cross_attn_v_proj(encoder_output)
        cross_attn_output = self.cross_attention(q, k, v)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.layer_norm2(x + cross_attn_output)

        # Feed-Forward Network
        ff_output = self.ff(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm3(x + ff_output)

        return x


class Encoder(nn.Module):

    def __init__(self, seq_length: int, vocab_size: int, embedding_dim: int, d_model: int, n_layer: int, num_heads: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attention_blocks = nn.ModuleList(
            EncoderBlock(d_model, embedding_dim, num_heads) for _ in range(n_layer)
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


class Decoder(nn.Module):

    def __init__(self, seq_length: int, vocab_size: int, embedding_dim: int, d_model: int, n_layer: int, num_heads: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attention_blocks = nn.ModuleList(
            DecoderBlock(d_model, embedding_dim, num_heads) for _ in range(n_layer)
        )
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input_seq: torch.Tensor, encoder_output: torch.Tensor):
        # TODO: pad
        token_embedding_matricx = self.token_embedding(input_seq)
        pos_embedding = get_positional_embedding(self.seq_length, self.d_model)
        input_embedding = (token_embedding_matricx + pos_embedding) * math.sqrt(
            self.embedding_dim)  # times this to control the value range
        output = None
        for layer in self.attention_blocks:
            output = layer(input_embedding, encoder_output)
        output = self.output_proj(output)
        logits = F.softmax(output, dim=-1)


        return logits


class Transformer(nn.Module):
    def __init__(self,
                 seq_length: int,
                 vocab_size: int,
                 embedding_dim: int,
                 d_model: int,
                 n_layer: int,
                 num_heads: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(
            seq_length=seq_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            d_model=d_model,
            n_layer=n_layer,
            num_heads=num_heads
        )

        self.decoder = Decoder(
            seq_length=seq_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            d_model=d_model,
            n_layer=n_layer,
            num_heads=num_heads
        )

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor):
        # src_seq: [batch_size, src_seq_length]
        # tgt_seq: [batch_size, tgt_seq_length]

        # 1. 通过encoder编码输入序列
        encoder_output = self.encoder(src_seq)

        # 2. 将encoder的输出和目标序列输入decoder
        output = self.decoder(tgt_seq, encoder_output)

        return output




if __name__ == "__main__":
    d_model = embedding_dim = 128
    num_heads = 4
    batch_size = 1
    # d_model represents the size of the hidden dimension,
    # embedding_dim represent the size of the the encoded tensor
    vocab_size = 1000
    seq_length = 32
    n_layer = 6
    # input_sqe = torch.tensor([torch.arange(seq_length), torch.arange(seq_length)])
    # input_seq = torch.randint(0, 100, size=(batch_size, seq_length))
    input_seq = torch.arange(seq_length)
    encoder = Encoder(seq_length=seq_length, vocab_size=vocab_size, embedding_dim=embedding_dim, d_model=d_model,
                      n_layer=n_layer, num_heads=num_heads)
    encoded_tensor = encoder(input_seq)
    print(encoded_tensor.shape) # torch.Size([32, 128])

    target_seq = torch.arange(start=1, end=seq_length+1) # 应该是input_seq right shift的了


    decoder = Decoder(seq_length=seq_length, vocab_size=vocab_size, embedding_dim=embedding_dim, d_model=d_model,
                      n_layer=n_layer, num_heads=num_heads)

    logis = decoder(target_seq, encoded_tensor) # torch.Size([32, 1000])

    print(f"logis:\n{logis.shape}")


    # transformer type
    src_seq = torch.arange(seq_length)
    tgt_seq = torch.arange(start=1, end=seq_length + 1)  # 目标序列右移一位

    transformer = Transformer(
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        d_model=d_model,
        n_layer=n_layer,
        num_heads=num_heads
    )

    # 前向传播
    output = transformer(src_seq, tgt_seq)
    print(f"Transformer output shape: {output.shape}")  # 应该是 [32, 1000]

