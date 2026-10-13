"""
PURPOSE: Containing the source code for training, validating, testing, and saving your model. The model should be imported from modules.py and the data loader should be imported from dataset.py. Make sure to plot the losses and metrics during training.
"""

import torch
import modules
import dataset

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    total_loss = 0.0

    for epoch in range(num_epochs):
        for images, masks, _ in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()   # Reset/clean
            loss.backward()         # Backpropagation
            optimizer.step()        # Update weights/gradient descent

            total_loss += loss.item()





ROOT = r"Dataset\keras_slices_data"
IMAGE_SHAPE = (256, 128) # Got it from looking inside the data

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")


    train_dataloader, validate_dataloader, test_dataloader, num_classes = dataset.get_dataloaders(ROOT, 8)

    model = modules.BasicUNet2D(1, 32, num_classes)

