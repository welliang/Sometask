# main.py

import argparse
import torch
from data_processing import download_and_prepare_data, prepare_dataloader
from training import train_model
from evaluation import evaluate_model
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Graph Similarity Search Project")
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode to run: train or evaluate')
    parser.add_argument('--dataset', default='AIDS', help='Dataset name (default: AIDS)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    set_seed(args.seed)

    # Download and prepare the dataset
    dataset = download_and_prepare_data(args.dataset)

    num_classes = dataset.num_labels  # 获取类别数量

    if args.mode == 'train':
        train_dataloader = prepare_dataloader(dataset, mode='train', batch_size=args.batch_size)
        # 初始化模型
        from models import GraphClassificationModel
        model = GraphClassificationModel(num_classes=num_classes, embedding_dim=args.embedding_dim)
        # 训练模型
        train_model(model, train_dataloader, epochs=args.epochs, learning_rate=args.learning_rate)
    elif args.mode == 'evaluate':
        test_dataloader = prepare_dataloader(dataset, mode='test', batch_size=args.batch_size)
        # 初始化模型
        from models import GraphClassificationModel
        model = GraphClassificationModel(num_classes=num_classes, embedding_dim=args.embedding_dim)
        # 加载训练模型参数
        model.load_state_dict(torch.load('model.pth'))
        # 评估模型
        evaluate_model(model, test_dataloader)

if __name__ == '__main__':
    main()
