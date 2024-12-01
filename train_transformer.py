import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformer import Transformer


# SimpleSequenceDataset类保持不变
class SimpleSequenceDataset(Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data = []

        for _ in range(num_samples):
            start = np.random.randint(0, 100)
            seq = [start]
            for i in range(seq_length - 1):
                next_num = seq[-1] + np.random.randint(1, 6)
                seq.append(next_num)
            self.data.append(seq)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        src_seq = torch.tensor(seq[:-1], dtype=torch.long)  # 修改为long类型
        tgt_seq = torch.tensor(seq[1:], dtype=torch.long)  # 修改为long类型
        return src_seq, tgt_seq


# train函数保持不变
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (src_seq, tgt_seq) in enumerate(train_loader):
        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)

        optimizer.zero_grad()
        output = model(src_seq, tgt_seq)

        output = output.view(-1, output.size(-1))
        tgt_seq = tgt_seq.view(-1)

        loss = criterion(output, tgt_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)


def main():
    # 修改后的参数
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    seq_length = 16  # 减小序列长度
    vocab_size = 200  # 减小词汇表大小
    embedding_dim = 64  # 减小嵌入维度
    d_model = 64  # 确保与embedding_dim相同
    n_layer = 2  # 减少层数
    num_heads = 8  # 确保能被d_model整除
    batch_size = 4
    num_epochs = 10
    num_samples = 1000

    # 创建数据集和数据加载器
    dataset = SimpleSequenceDataset(seq_length + 1, num_samples)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = Transformer(
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        d_model=d_model,
        n_layer=n_layer,
        num_heads=num_heads
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 略微增加学习率

    # 训练循环
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')


if __name__ == "__main__":
    main()
