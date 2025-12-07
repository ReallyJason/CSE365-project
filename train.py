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

# Configuration
MODEL_SAVE_PATH = "saved_models"
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0008  # Slightly reduced for more stable training with stronger regularization
WEIGHT_DECAY = 2e-4  # Further increased L2 regularization to prevent overfitting
GRADIENT_CLIP = 1.0  # Clip gradients to prevent exploding gradients and improve stability
EARLY_STOPPING_PATIENCE = 8  # Reduced patience to stop when validation plateaus/declines


class GenreClassifier(nn.Module):
    """
    CNN for music genre classification.
    
    Architecture: 2 conv blocks (conv + pool + batch norm) -> 2 fully connected layers.
    Input format: (N, C, H, W) - channels-first as provided by preprocess.py
    """
    
    def __init__(self, num_channels: int, height: int, width: int, num_classes: int):
        super(GenreClassifier, self).__init__()
        
        # First convolutional block: extract low-level features
        # Reduced filters to decrease model capacity
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.35)  # Further increased spatial dropout
        
        # Second convolutional block: extract higher-level features
        # Reduced filters to decrease model capacity
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)  # Reduced from 64 to 48
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.dropout2 = nn.Dropout2d(0.35)  # Further increased spatial dropout
        
        # Calculate flattened size after two pooling operations (each halves dimensions)
        flattened_height = height // 4  # After two pool operations
        flattened_width = width // 4
        flatten_size = 48 * flattened_height * flattened_width  # Updated for 48 filters
        
        # Fully connected layers for classification
        # Further reduced FC layer size to reduce model capacity and prevent overfitting
        self.fc1 = nn.Linear(flatten_size, 64)  # Reduced from 96 to 64
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.7)  # Further increased dropout to prevent overfitting
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First conv block: extract features and reduce spatial dimensions
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)  # Dropout after conv to prevent overfitting
        
        # Second conv block: extract higher-level features
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)  # Dropout after conv to prevent overfitting
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classification layers with increased dropout for stronger regularization
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
    Train the model with early stopping and learning rate reduction.
    
    Saves the best model based on validation loss and stops early if no improvement.
    """
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Setup optimizer with weight decay (L2 regularization) to prevent overfitting
    # Using fixed learning rate (no scheduler)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Track validation accuracy plateau for potential manual LR reduction
    val_acc_plateau_counter = 0
    best_val_acc_seen = 0.0
    
    # Track training history and best model
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("\nStarting training...")
    print(f"Device: {device}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE} (fixed)")
    print(f"Weight decay (L2): {WEIGHT_DECAY} (helps prevent overfitting)")
    print(f"Max epochs: {epochs}, Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Clip gradients to prevent exploding gradients and improve training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            # Track accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Record history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print epoch results
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            best_model_path = os.path.join(MODEL_SAVE_PATH, "best_model_pytorch.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, best_model_path)
            print(f"  ✓ Saved best model (val_acc: {val_accuracy * 100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
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
    
    # Collect predictions and labels
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate overall test metrics
    test_loss /= len(test_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    test_accuracy = (all_predictions == all_labels).mean()
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy * 100:.2f}%")
    
    # Print per-genre accuracy breakdown
    print("\nPer-genre accuracy:")
    for i, genre in enumerate(mapping):
        mask = all_labels == i
        if mask.sum() > 0:
            correct = (all_predictions[mask] == i).sum()
            total = mask.sum()
            class_accuracy = correct / total
            print(f"  {genre}: {class_accuracy * 100:.2f}% ({correct}/{total} correct)")


def save_model(model: nn.Module, mapping: List[str], num_channels: int, height: int, width: int) -> None:
    """Save the trained model state and genre mapping for later inference."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Save model checkpoint with metadata needed for loading
    model_path = os.path.join(MODEL_SAVE_PATH, "final_model_pytorch.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_channels': num_channels,
        'height': height,
        'width': width,
        'num_classes': len(mapping),
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save genre mapping for label interpretation
    mapping_path = os.path.join(MODEL_SAVE_PATH, "genre_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    print(f"Genre mapping saved to: {mapping_path}")


def main():
    """Main training pipeline: load data, train model, evaluate, and save."""
    print("=" * 60)
    print("PyTorch Training Pipeline for Music Genre Classification")
    print("=" * 60)
    
    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU not available, using CPU")
    print()
    
    # Load data (automatically processes from JSON if needed)
    # Data comes in channels-first format: (N, C, H, W)
    X_train, X_val, X_test, y_train, y_val, y_test, mapping = load_data(device=device)
    
    # Extract input dimensions from training data
    num_channels, height, width = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    num_classes = len(mapping)
    
    print(f"\nModel configuration:")
    print(f"  Input shape: (C={num_channels}, H={height}, W={width})")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {mapping}")
    
    # Create DataLoaders for batching
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model and move to device
    model = GenreClassifier(num_channels, height, width, num_classes).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train the model
    print("\n" + "=" * 60)
    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS)
    
    # Evaluate on test set and save model
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    evaluate_model(model, test_loader, mapping, device)
    
    save_model(model, mapping, num_channels, height, width)
    
    # Print training summary
    if history['val_accuracy']:
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        print(f"\nTraining Summary:")
        print(f"  Epochs completed: {len(history['loss'])}")
        print(f"  Training accuracy: {final_train_acc * 100:.2f}%")
        print(f"  Validation accuracy: {final_val_acc * 100:.2f}%")
        print("Training complete!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
