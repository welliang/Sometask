import torch
import torch.nn as nn
import torch.optim as optim


class G2GSimilarityNetwork(nn.Module):
    def __init__(self, embedding_dim=64):
        super(G2GSimilarityNetwork, self).__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, emb1, emb2):
        emb1 = self.fc(emb1)
        emb2 = self.fc(emb2)
        return torch.cosine_similarity(emb1, emb2, dim=1)


def train(model, embeddings, num_epochs=10, learning_rate=0.001, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    num_graphs = len(embeddings)
    embeddings_tensor = torch.tensor(embeddings).float()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i in range(0, num_graphs, batch_size):
            batch_emb = embeddings_tensor[i:i + batch_size]

            for j in range(batch_emb.size(0)):
                for k in range(j + 1, batch_emb.size(0)):
                    emb1 = batch_emb[j].unsqueeze(0)
                    emb2 = batch_emb[k].unsqueeze(0)

                    optimizer.zero_grad()
                    output = model(emb1, emb2)
                    target = calculate_similarity(emb1, emb2)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

        avg_loss = total_loss / (num_graphs * (num_graphs - 1) / 2)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


def calculate_similarity(emb1, emb2):
    return torch.cosine_similarity(emb1, emb2, dim=1)