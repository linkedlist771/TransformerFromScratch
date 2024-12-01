"""
gpt.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import get_positional_embedding, FeedForward, EncoderBlock


class GPT(nn.Module):
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
            EncoderBlock(d_model, embedding_dim, num_heads, True) for _ in range(n_layer)
        )
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input_seq: torch.Tensor):
        # TODO: pad
        token_embedding_matricx = self.token_embedding(input_seq)
        pos_embedding = get_positional_embedding(self.seq_length, self.d_model)
        input_embedding = (token_embedding_matricx + pos_embedding) * math.sqrt(
            self.embedding_dim)  # times this to control the value range
        output = self.attention_blocks[0](input_embedding)
        for layer in self.attention_blocks[1:]:
            output = layer(output)

        #
        output = self.output_proj(output)

        return F.softmax(output, dim=-1)




if __name__ == "__main__":
    seq_length = 32
    vocab_size = 1000
    d_model = embedding_dim = 128
    head_nums = 8
    n_layers = 6
    input_seq = torch.arange(seq_length)
    gpt = GPT(seq_length=seq_length, vocab_size=vocab_size, embedding_dim=embedding_dim, d_model=d_model, n_layer=n_layers, num_heads=head_nums)
    res = gpt(input_seq)
    print(f"res:{res.shape}")
    print(f"res:{torch.argmax(res, dim=-1)}") # 得到了实际的token



