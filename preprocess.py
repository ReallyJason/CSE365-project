#!/usr/bin/env python3.12
"""
PyTorch data preprocessing for music genre classification.

Handles the complete pipeline from data.json to ready-to-use PyTorch tensors.
Automatically processes from JSON if preprocessed data doesn't exist.
"""
import json
import os
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Configuration constants
JSON_PATH = "data.json"
PREPROCESSED_DATA_PATH = "preprocessed_data.npz"
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1


def _load_from_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load raw MFCC features, labels, and genre mapping from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["mfcc"]), np.array(data["labels"]), data["mapping"]


def _prepare_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare and split data: add channel dimension, normalize, and split into train/val/test.
    
    Returns data in (N, H, W, C) format before PyTorch conversion.
    """
    # Add channel dimension if missing: (N, H, W) -> (N, H, W, C)
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    
    # Normalize features to [0, 1] range for better training stability
    X_min, X_max = X.min(), X.max()
    if X_max > X_min:
        X = (X - X_min) / (X_max - X_min)
    
    # Split data: 10% test, then 80% train / 20% val from remaining 90%
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=42, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _save_to_disk(X_train, X_val, X_test, y_train, y_val, y_test, mapping):
    """Save preprocessed numpy arrays to compressed .npz file for faster loading."""
    np.savez_compressed(
        PREPROCESSED_DATA_PATH,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        mapping=mapping
    )


def _load_from_disk() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load preprocessed data from disk."""
    data = np.load(PREPROCESSED_DATA_PATH, allow_pickle=True)
    return (
        data["X_train"], data["X_val"], data["X_test"],
        data["y_train"], data["y_val"], data["y_test"],
        data["mapping"].tolist()
    )


def load_data(device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Load data for PyTorch training in channels-first format (N, C, H, W).
    
    Automatically processes from data.json if preprocessed data doesn't exist.
    All tensors are ready for DataLoader creation.
    
    Args:
        device: PyTorch device (CPU/GPU). If None, tensors remain on CPU.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, mapping
    """
    # Process from JSON if preprocessed data doesn't exist
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        print("Preprocessed data not found. Processing from data.json...")
        X, y, mapping = _load_from_json(JSON_PATH)
        X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(X, y)
        _save_to_disk(X_train, X_val, X_test, y_train, y_val, y_test, mapping)
        print(f"✓ Data preprocessed and saved to {PREPROCESSED_DATA_PATH}")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, mapping = _load_from_disk()
    
    # Convert numpy arrays to PyTorch tensors in channels-first format: (N, H, W, C) -> (N, C, H, W)
    X_tensors = [torch.from_numpy(X).permute(0, 3, 1, 2).float() for X in [X_train, X_val, X_test]]
    y_tensors = [torch.from_numpy(y).long() for y in [y_train, y_val, y_test]]
    
    # Move all tensors to specified device if provided
    if device is not None:
        X_tensors = [X.to(device) for X in X_tensors]
        y_tensors = [y.to(device) for y in y_tensors]
    
    return X_tensors[0], X_tensors[1], X_tensors[2], y_tensors[0], y_tensors[1], y_tensors[2], mapping


if __name__ == "__main__":
    """Standalone preprocessing: process from JSON and save to disk."""
    print("Processing data from JSON...")
    X, y, mapping = _load_from_json(JSON_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(X, y)
    _save_to_disk(X_train, X_val, X_test, y_train, y_val, y_test, mapping)
    print(f"✓ Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
