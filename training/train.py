import sys
import os

# Ajouter le chemin racine du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.convlstm import ConvLSTM
from download_data import get_moving_mnist_dataset

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 1  # Niveaux de gris (Moving MNIST)
hidden_dim = [64, 64]  # Architecture du papier
kernel_size = 3
num_layers = len(hidden_dim)
learning_rate = 0.001
num_epochs = 10
batch_size = 16

# Chargement des données
train_loader = get_moving_mnist_dataset(batch_size=batch_size, train=True)

# Initialisation du modèle
model = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers).to(device)

# Fonction de perte et optimiseur
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

# Initialisation de TensorBoard
writer = SummaryWriter(log_dir="runs/convLSTM_experiment")

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(device)

        # Séparer les 10 premières frames (entrée) et les 10 dernières (cible)
        input_seq = inputs[:, :10, :, :, :] 
        target_seq = inputs[:, 10:, :, :, :]

        print(f"Input shape: {input_seq.shape}, Target shape: {target_seq.shape}")  # Vérification

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_seq)
        outputs = torch.sigmoid(outputs)  # Activation pour BCE Loss

        # Calcul de la perte
        loss = criterion(outputs, target_seq)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Ajouter la perte de ce batch à TensorBoard
        writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

    # Afficher la perte moyenne par époque
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    # Ajouter la perte moyenne de l'époque à TensorBoard
    writer.add_scalar("Loss/Epoch", avg_loss, epoch)

# Sauvegarde du modèle
torch.save(model.state_dict(), "output/convlstm.pth")
print("Modèle sauvegardé dans output/convlstm.pth")

# Fermer TensorBoard
writer.close()
