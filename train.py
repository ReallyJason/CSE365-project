#!/usr/bin/env python3.12
"""
PyTorch training script for music genre classification.
"""
import json
import os
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_data

# These are the settings we use for training
MODEL_SAVE_PATH = "saved_models"
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0008  # Learning rate for the optimizer
WEIGHT_DECAY = 2e-4  # L2 regularization to help prevent overfitting
GRADIENT_CLIP = 1.0  # Maximum gradient value to prevent exploding gradients
EARLY_STOPPING_PATIENCE = 8  # How many epochs to wait before stopping if no improvement


class GenreClassifier(nn.Module):
    """
    CNN for music genre classification.
    
    This network has 2 convolutional blocks followed by 2 fully connected layers.
    The input should be in channels-first format: (N, C, H, W)
    """
    
    def __init__(self, num_channels: int, height: int, width: int, num_classes: int):
        super(GenreClassifier, self).__init__()
        
        # First convolutional block - extracts basic features from the audio
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.35)
        
        # Second convolutional block - extracts more complex features
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.dropout2 = nn.Dropout2d(0.35)
        
        # Figure out how big the flattened layer needs to be
        # Each pooling operation cuts the size in half, so we do it twice
        flattened_height = height // 4
        flattened_width = width // 4
        flatten_size = 48 * flattened_height * flattened_width
        
        # Fully connected layers that do the actual classification
        self.fc1 = nn.Linear(flatten_size, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First convolutional block
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Flatten the data so we can use fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers for classification
        x = torch.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> Dict[str, List[float]]:
    """
    Train the model with early stopping.
    
    This function trains the model and saves the best one based on validation loss.
    It will stop early if the model stops improving.
    """
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Keep track of validation accuracy for monitoring
    val_acc_plateau_counter = 0
    best_val_acc_seen = 0.0
    
    # Keep track of training progress and the best model we've seen
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("")
    print("Starting training...")
    print("Device: " + str(device))
    print("Batch size: " + str(BATCH_SIZE))
    print("Learning rate: " + str(LEARNING_RATE))
    print("Weight decay: " + str(WEIGHT_DECAY))
    print("Max epochs: " + str(epochs))
    print("Early stopping patience: " + str(EARLY_STOPPING_PATIENCE))
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase - go through all training data
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Loop through each batch of training data
        for batch_x, batch_y in train_loader:
            # Reset gradients before computing new ones
            optimizer.zero_grad()
            
            # Get predictions from the model
            outputs = model(batch_x)
            
            # Calculate how wrong we are
            loss = criterion(outputs, batch_y)
            
            # Backpropagate to update weights
            loss.backward()
            
            # Clip gradients to prevent them from getting too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            
            # Update the model weights
            optimizer.step()
            
            # Keep track of how many we got right
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Calculate average loss and accuracy for this epoch
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase - check how well we're doing on validation data
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Don't update weights during validation
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Keep track of validation metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Save these results for later
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print what happened this epoch
        print("Epoch " + str(epoch + 1) + "/" + str(epochs))
        print("  Train Loss: " + str(round(train_loss, 4)) + ", Train Acc: " + str(round(train_accuracy * 100, 2)) + "%")
        print("  Val Loss: " + str(round(val_loss, 4)) + ", Val Acc: " + str(round(val_accuracy * 100, 2)) + "%")
        
        # Check if this is the best model we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save this model as the best one
            best_model_path = os.path.join(MODEL_SAVE_PATH, "best_model_pytorch.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, best_model_path)
            print("  Saved best model (val_acc: " + str(round(val_accuracy * 100, 2)) + "%)")
        else:
            # We didn't improve, so increment the patience counter
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("")
                print("Early stopping triggered after " + str(epoch + 1) + " epochs")
                print("Best validation loss: " + str(round(best_val_loss, 4)))
                print("Best validation accuracy: " + str(round(best_val_accuracy * 100, 2)) + "%")
                model.load_state_dict(best_model_state)
                break
        
        print("-" * 60)
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    mapping: List[str],
    device: torch.device
) -> None:
    """Evaluate the model on the test set and print per-genre accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Go through all test data and collect predictions
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate overall test metrics
    test_loss = test_loss / len(test_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    test_accuracy = (all_predictions == all_labels).mean()
    
    print("")
    print("Test Results:")
    print("  Loss: " + str(round(test_loss, 4)))
    print("  Accuracy: " + str(round(test_accuracy * 100, 2)) + "%")
    
    # Print accuracy for each genre separately
    print("")
    print("Per-genre accuracy:")
    for i, genre in enumerate(mapping):
        mask = all_labels == i
        if mask.sum() > 0:
            correct = (all_predictions[mask] == i).sum()
            total = mask.sum()
            class_accuracy = correct / total
            print("  " + genre + ": " + str(round(class_accuracy * 100, 2)) + "% (" + str(correct) + "/" + str(total) + " correct)")


def save_model(model: nn.Module, mapping: List[str], num_channels: int, height: int, width: int) -> None:
    """Save the trained model state and genre mapping for later inference."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Save the model with all the info we need to load it later
    model_path = os.path.join(MODEL_SAVE_PATH, "final_model_pytorch.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_channels': num_channels,
        'height': height,
        'width': width,
        'num_classes': len(mapping),
    }, model_path)
    print("")
    print("Model saved to: " + model_path)
    
    # Save the genre mapping so we know which number means which genre
    mapping_path = os.path.join(MODEL_SAVE_PATH, "genre_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    print("Genre mapping saved to: " + mapping_path)


def main():
    """Main training pipeline: load data, train model, evaluate, and save."""
    print("=" * 60)
    print("PyTorch Training Pipeline for Music Genre Classification")
    print("=" * 60)
    
    # Check if we have a GPU available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU detected: " + str(torch.cuda.get_device_name(0)))
    else:
        print("GPU not available, using CPU")
    print("")
    
    # Load the data - this will process it from JSON if needed
    # The data comes in channels-first format: (N, C, H, W)
    X_train, X_val, X_test, y_train, y_val, y_test, mapping = load_data(device=device)
    
    # Figure out the dimensions of our input data
    num_channels = X_train.shape[1]
    height = X_train.shape[2]
    width = X_train.shape[3]
    num_classes = len(mapping)
    
    print("")
    print("Model configuration:")
    print("  Input shape: (C=" + str(num_channels) + ", H=" + str(height) + ", W=" + str(width) + ")")
    print("  Number of classes: " + str(num_classes))
    print("  Classes: " + str(mapping))
    
    # Create data loaders that will give us batches of data
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Create the model and move it to the right device (GPU or CPU)
    model = GenreClassifier(num_channels, height, width, num_classes).to(device)
    
    print("")
    print("Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: " + str(total_params))
    
    # Train the model
    print("")
    print("=" * 60)
    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS)
    
    # Save the training history so we can look at it later
    history_path = os.path.join(MODEL_SAVE_PATH, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print("Training history saved to " + history_path)
    
    # Test the model on the test set and save it
    print("")
    print("=" * 60)
    print("Evaluating on test set...")
    evaluate_model(model, test_loader, mapping, device)
    
    save_model(model, mapping, num_channels, height, width)
    
    # Print a summary of what happened
    if history['val_accuracy']:
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        print("")
        print("Training Summary:")
        print("  Epochs completed: " + str(len(history['loss'])))
        print("  Training accuracy: " + str(round(final_train_acc * 100, 2)) + "%")
        print("  Validation accuracy: " + str(round(final_val_acc * 100, 2)) + "%")
        print("Training complete!")
    
    print("")
    print("=" * 60)


if __name__ == "__main__":
    main()

