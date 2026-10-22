"""
PURPOSE: Containing the source code for training, validating, testing, and saving your model. The model should be imported from modules.py and the data loader should be imported from dataset.py. Make sure to plot the losses and metrics during training.
"""

import time

from torch.utils.data import DataLoader, Subset

import torch
import torch.nn.functional as F
import modules
import dataset

import argparse

def calculate_dice_loss(predictions, labels, eps=1e-6):
    """
    Multi-class dice loss. 
    Prediction and label are of form (B, C, H, W)
    Label targets are one-hot
    """
    prediction_probs = torch.softmax(predictions, dim=1)

    numerator = (prediction_probs * labels).sum(dim=(0, 2, 3))
    denominator = prediction_probs.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))

    sum_dice = (2*numerator + eps) / (denominator + eps) # Per class dice
    multi_class_dice_loss = 1 - sum_dice.mean() # mean() for the dividing by |K| classes

    return multi_class_dice_loss

def calculate_dice_components(predictions, labels, num_classes):
    """
    Returns the overlap and total counts, so you can calculate dice coefficient and IoU 
    Prediction and label are of form (B, C, H, W)
    Label targets are one-hot

    Dice coefficient (DSC) per class/channel (1.0 = perfect match, 0.0 means no overlap)
    """
    prediction_answers = predictions.argmax(dim=1) # Picks the class with the highest score (and stores the classID as the entries)
    predictions_onehot = F.one_hot(prediction_answers, num_classes).permute(0, 3, 1, 2).float() # Convert classIDs to one-hot entries, also (B, 256, 128, 6) -> (B, 6, 256, 128)
    
    overlap = (predictions_onehot * labels).sum(dim=(0, 2, 3)) # 1 if both are 1 (1*1), 0 otherwise
    total = predictions_onehot.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))

    return overlap, total

def train_model(model, train_dataloader, validate_dataloader, optimizer, num_epochs, model_save_path, device, num_classes, eps=1e-6):
    model = model.to(device)
    training_history = {"Train Loss": [], "Validate Loss": []}
    best_validation_loss = float("inf")

    step = 0
    for epoch in range(num_epochs):

        # Step 1: Train
        train_loss = 0.0
        model.train()
        for images, masks, _ in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = calculate_dice_loss(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()   # Reset/clean
            loss.backward()         # Backpropagation

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if step % 100 == 0 or grad_norm > 10:
                print(f"step {step} loss {loss.item():.3f} grad_norm {grad_norm:.2f} "
                    f"max_logit {outputs.abs().max().item():.1f}")

            optimizer.step()        # Update weights/gradient descent
            step += 1

            train_loss += loss.item()

        # Step 2: Validate set evaluation
        overlap = torch.zeros(num_classes, device=device)
        total = torch.zeros(num_classes, device=device)

        validate_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks, _ in validate_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                validate_loss += calculate_dice_loss(outputs, masks).item()

                new_overlap, new_total = calculate_dice_components(outputs, masks, num_classes)
                
                overlap += new_overlap
                total += new_total

        # Step 3: Record training and validation losses
        avg_training_loss = train_loss / len(train_dataloader)
        avg_validate_loss = validate_loss / len(validate_dataloader)
        training_history["Train Loss"].append(avg_training_loss)
        training_history["Validate Loss"].append(avg_validate_loss)

        print(f"epoch: {epoch+1}")
        print(f"    - train_loss: {train_loss / len(train_dataloader)}")
        print(f"    - validate_loss: {validate_loss / len(validate_dataloader)}")

        # Calculate dice coefficient and IoU per class
        dice = (2*overlap + eps) / (total + eps)
        IoU = (overlap + eps) /(total - overlap + eps)
        print("-------- per-class -----------")
        for c in range(num_classes):
            if total[c] == 0:
                print(f"    - {c}: dice coefficient: N/A | IoU: N/A (absent from validation labels and predictions)")
                continue

            print(f"    - {c}: dice coefficient: {dice[c].item()} | IoU: {IoU[c].item()}") # Ideally want both to be 1 (perfect match). 0 means no overlap.
        print("-----------------------------")
        
        
        # Step 4: Store the model if it's the best seen so far
        if (best_validation_loss > avg_validate_loss):
            print(f"    - Saving for {avg_validate_loss}")
            best_validation_loss = avg_validate_loss
            torch.save(model.state_dict(), model_save_path)

    return training_history

def test_model(model, test_dataloader, num_classes, model_save_path, device, eps=1e-6):
    """
    Load model, and report per-class Dice and IoU over the whole test
    """
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)

    # Keep a tally/count of all overlap and total so that we can compute dice and IoU per-class at the end
    overlap = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)
    
    model.eval()
    with torch.no_grad():
        for images, masks, _ in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            new_overlap, new_total = calculate_dice_components(outputs, masks, num_classes)

            overlap += new_overlap
            total += new_total

    # Calculate dice coefficient and IoU per class
    dice = (2*overlap + eps) / (total + eps)
    IoU = (overlap + eps) /(total - overlap + eps)
    for c in range(num_classes):
        print(f"{c}: dice coefficient: {dice[c].item()} | IoU: {IoU[c].item()}") # Ideally want both to be 1 (perfect match). 0 means no overlap.



############# HYPER-PARAMETERS #############
learning_rate = 1e-4
batch_size = 8
num_epochs = 10 # 50 for real
base_channels = 8 # 32 for real
############################################
ROOT = r"Dataset\keras_slices_data"
UNET_MODEL_PATH = "Basic-U-Net.pt"
CAN_MODEL_PATH = "Basic-2D-CAN.pt"

def parse_args():
    parser = argparse.ArgumentParser()

    # File IO
    parser.add_argument("--data_dir", default = ROOT)
    parser.add_argument("--save_path", default = CAN_MODEL_PATH)

    # Hyper parameters
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument("--base_channels", type=int, default=base_channels)

    return parser.parse_args()



if __name__ == "__main__":
    # Taking in arguments
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    train_dataloader, validate_dataloader, test_dataloader, num_classes = dataset.get_dataloaders(args.data_dir, args.batch_size)
    print("Finished data loading")

    ### For testing purposes ###
    train_small = DataLoader(
        Subset(train_dataloader.dataset, range(2000, 2064)),
                         batch_size=8, shuffle=True)
    validate_small = DataLoader(
        Subset(validate_dataloader.dataset, range(100, 132)),
                        batch_size=8, shuffle=False)
    ############################

    model = modules.Basic2DCAN(1, args.base_channels, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"params: lr={args.lr}, batch_size={batch_size}, num_epoch={num_epochs}, base_channels={base_channels}")

    start_time = time.time()
    training_history = train_model(model, train_dataloader, validate_dataloader, optimizer, args.num_epochs, args.save_path, device, num_classes)
    end_time = time.time()

    time_elapsed = end_time - start_time
    print(f"Total training time: {time_elapsed/60} minutes")

    test_model(model, test_dataloader, num_classes, args.save_path, device)
