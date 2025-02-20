import torch
from torchvision.datasets import MovingMNIST
from torch.utils.data import DataLoader

def get_moving_mnist_dataset(root_dir='data', batch_size=32, num_workers=2, train=True):
    """
    Charge MovingMNIST avec un DataLoader sans appliquer ToTensor().
    """

    dataset = MovingMNIST(
        root=root_dir,
        split='train' if train else 'test',
        download=True
    ) 

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
        print(f"Batch {batch_idx}: Inputs shape: {inputs.shape}")
        break  # Affiche uniquement le premier batch
