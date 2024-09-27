import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        x = F.relu(self.fc1(diff))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def generate_pairs(embeddings):
    num_graphs = len(embeddings)
    pairs = []
    labels = []
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            pairs.append((embeddings[i], embeddings[j]))
            # 使用余弦相似度计算
            similarity = np.dot(embeddings[i].flatten(), embeddings[j].flatten()) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            labels.append(similarity)
    return pairs, labels


def train(siamese_model, embeddings, num_epochs=10, batch_size=32, learning_rate=0.001):
    pairs, labels = generate_pairs(embeddings)

    dataset = TensorDataset(
        torch.tensor([p[0] for p in pairs]).float(),
        torch.tensor([p[1] for p in pairs]).float(),
        torch.tensor(labels).float()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(siamese_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        siamese_model.train()
        total_loss = 0.0

        for batch_x1, batch_x2, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = siamese_model(batch_x1, batch_x2).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    return siamese_model