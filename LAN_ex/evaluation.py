# evaluation.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batched_graph, labels in tqdm(dataloader, desc="Evaluating"):
            outputs = model(batched_graph)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
