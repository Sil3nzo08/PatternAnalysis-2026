"""
PURPOSE: Containing the source code for training, validating, testing, and saving your model. The model should be imported from modules.py and the data loader should be imported from dataset.py. Make sure to plot the losses and metrics during training.
"""

import torch
import torch.nn.functional as F
import modules
import dataset

def calculate_dice_loss(predictions, labels, eps=1e-6):
    """
    Multi-class dice loss. 
    Prediction and label are of form (B, C, H, W)
    Label targets are one-hot
    """
    prediction_probs = torch.softmax(predictions, dim=1)

    numerator = (prediction_probs * labels).sum(dim=(0, 2, 3))
    denominator = prediction_probs.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))

    sum = (2*numerator + eps) / (denominator + eps) # Per class dice
    multi_class_dice_loss = 1 - sum.mean() # mean() for the dividing by |K| classes

    return multi_class_dice_loss

def calculate_dice_per_channel(predictions, labels, num_classes, eps=1e-6):
    """
    Dice coefficient (DSC) per class/channel (1.0 = perfect match, 0.0 means no overlap)
    Prediction and label are of form (B, C, H, W)
    Label targets are one-hot
    """
    prediction_answers = predictions.argmax(dim=1) # Picks the class with the highest score (and stores the classID as the entries)
    predictions_onehot = F.one_hot(prediction_answers, num_classes).permute(0, 3, 1, 2).float() # Convert classIDs to one-hot entries, also (B, 256, 128, 6) -> (B, 6, 128, 256)
    

    numerator = (predictions_onehot * labels).sum(dim=(0, 2, 3)) # 1 if both are 1 (1*1), 0 otherwise
    denominator = predictions_onehot.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))
    dice_coefficient = (2*numerator + eps) / (denominator + eps) # Per class dice

    return dice_coefficient # A tensor with the 6 dice cefficients [DSC for class 0, DSC for class 1, ...]






def train_model(model, train_dataloader, validate_dataloader, optimizer, num_epochs, num_classes, device):
    model = model.to(device)

    
    for epoch in range(num_epochs):

        # Step 1: Train
        total_train_loss = 0.0
        model.train()
        for images, masks, _ in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            train_loss = calculate_dice_loss(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()   # Reset/clean
            train_loss.backward()         # Backpropagation
            optimizer.step()        # Update weights/gradient descent

            total_train_loss += train_loss.item()

        # Step 2: Validate set evaluation
        validate_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks, _ in validate_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                validate_loss += calculate_dice_loss(outputs, masks).item()
                per_class_DSC = calculate_dice_per_channel(outputs, masks, num_classes)

        print(f"epoch: {epoch + 1}/{num_epochs} | train: {total_train_loss / len(train_dataloader)} | validate: {validate_loss / len(validate_dataloader)}")
        print(per_class_DSC)








ROOT = r"Dataset\keras_slices_data"
IMAGE_SHAPE = (256, 128) # Got it from looking inside the data

from torch.utils.data import Subset, DataLoader

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    print("Here1")

    train_dataloader, validate_dataloader, test_dataloader, num_classes = dataset.get_dataloaders(ROOT, 8)

    print("Here2")

    train_small = DataLoader(
        Subset(train_dataloader.dataset, range(2000, 2064)),
                         batch_size=8, shuffle=True)
    val_small = DataLoader(
        Subset(validate_dataloader.dataset, range(100, 132)),
                        batch_size=8, shuffle=False)

    model = modules.BasicUNet2D(1, 8, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_small, val_small, optimizer, 10, num_classes, device)
