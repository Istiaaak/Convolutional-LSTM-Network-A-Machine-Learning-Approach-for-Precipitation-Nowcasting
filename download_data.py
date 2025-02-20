import torch
from torchvision.datasets import MovingMNIST
from torch.utils.data import DataLoader

def get_moving_mnist_dataset(root_dir='data', batch_size=16, num_workers=2, train=True):
    dataset = MovingMNIST(root=root_dir, split='train' if train else 'test', download=True)

    print(f"Dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader

# Test rapide si ce fichier est exécuté seul
if __name__ == "__main__":
    dataloader = get_moving_mnist_dataset()

    for batch_idx, inputs in enumerate(dataloader):
        print(f"Batch {batch_idx}: Full sequence shape: {inputs.shape}")  # Devrait être (batch_size, 20, 1, 64, 64)
        break
