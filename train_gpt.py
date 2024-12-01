import torch
import torch.nn as nn
from gpt import GPT


# 1. 准备一个简单的词表和训练数据
def create_vocab():
    # 简单词表：包含基本标点和常用词
    words = ["<pad>", "<unk>", ".", "!", "?", "the", "is", "a", "cat", "dog", "sits", "on", "mat"]
    return {word: idx for idx, word in enumerate(words)}, {idx: word for idx, word in enumerate(words)}


def encode_sentence(sentence, word2idx):
    # 将句子转换为token序列
    return [word2idx.get(word.lower(), word2idx["<unk>"]) for word in sentence.split()]


# 2. 训练设置
def train_gpt():
    # 模型参数
    seq_length = 32
    vocab_size = 13  # 词表大小
    d_model = embedding_dim = 128
    head_nums = 8
    n_layers = 6

    # 创建词表
    word2idx, idx2word = create_vocab()

    # 准备训练数据
    train_sentence = "The cat sits on the mat ."
    input_tokens = encode_sentence(train_sentence, word2idx)

    # 填充序列到指定长度
    while len(input_tokens) < seq_length:
        input_tokens.append(word2idx["<pad>"])

    # 创建输入tensor
    input_seq = torch.tensor(input_tokens)

    # 创建模型
    model = GPT(
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        d_model=d_model,
        n_layer=n_layers,
        num_heads=head_nums
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 前向传播
        output = model(input_seq)

        # 计算损失（这里我们预测下一个词）
        target = input_seq[1:]  # 移除第一个token
        output = output[:-1]  # 移除最后一个预测

        loss = criterion(output.view(-1, vocab_size), target)

        # 反向传播
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 生成一些文本
    # input_seq =
    with torch.no_grad():
        output = model(input_seq)
        predicted = torch.argmax(output, dim=-1)

        # 将预测的token转换回文字
        predicted_words = [idx2word[idx.item()] for idx in predicted]
        print("\n生成的文本:")
        print(" ".join(predicted_words))


if __name__ == "__main__":
    train_gpt()
