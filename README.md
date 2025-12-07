# Music Genre Classifier - Training and Preprocessing Guide

This README provides detailed documentation for the `preprocess.py` and `train.py` programs used in the Music Genre Classification project.

## Overview

- **`preprocess.py`**: Handles data preprocessing from raw JSON format to PyTorch-ready tensors
- **`train.py`**: Trains a CNN model for music genre classification using PyTorch

Both scripts work together to create a complete training pipeline from raw audio features to a trained model.

---

## Requirements

### Python Version
- Python 3.12 or compatible

### Dependencies
Install the required packages:

```bash
pip install numpy torch scikit-learn
```

Or if you have a requirements file:
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical operations and array handling
- `torch` - PyTorch for deep learning (automatically uses GPU if available)
- `scikit-learn` - For data splitting (train_test_split)

---

## Data Format

The preprocessing script expects a `data.json` file in the project root with the following structure:

```json
{
  "mfcc": [...],  // MFCC features array
  "labels": [...],  // Integer labels corresponding to genres
  "mapping": [...]  // List of genre names (string labels)
}
```

The preprocessed data is saved as `preprocessed_data.npz` for faster subsequent loading.

---

## Preprocessing (`preprocess.py`)

### Purpose
Converts raw MFCC features from JSON format into PyTorch tensors ready for training. Automatically handles normalization, data splitting, and tensor format conversion.

### Key Features
- **Automatic processing**: Automatically processes from JSON if preprocessed data doesn't exist
- **Data normalization**: Normalizes features to [0, 1] range for training stability
- **Train/Validation/Test split**: Automatically splits data (70% train, 20% validation, 10% test)
- **Stratified splitting**: Maintains class distribution across splits
- **Format conversion**: Converts to PyTorch channels-first format (N, C, H, W)

### Usage

#### Option 1: Standalone Preprocessing
Run preprocessing independently to generate `preprocessed_data.npz`:

```bash
python preprocess.py
```

This will:
1. Load data from `data.json`
2. Normalize features
3. Split into train/validation/test sets
4. Save to `preprocessed_data.npz`

#### Option 2: Automatic Preprocessing (Recommended)
The preprocessing happens automatically when you run `train.py`. If `preprocessed_data.npz` doesn't exist, it will be created from `data.json` automatically.

### Configuration Constants

You can modify these in `preprocess.py` if needed:

```python
JSON_PATH = "data.json"                    # Input JSON file
PREPROCESSED_DATA_PATH = "preprocessed_data.npz"  # Output file
VALIDATION_SPLIT = 0.2                     # 20% of training data for validation
TEST_SPLIT = 0.1                           # 10% of all data for testing
```

### Output
- **`preprocessed_data.npz`**: Compressed NumPy archive containing:
  - `X_train`, `X_val`, `X_test` - Feature arrays
  - `y_train`, `y_val`, `y_test` - Label arrays
  - `mapping` - Genre name mapping

---

## Training (`train.py`)

### Purpose
Trains a CNN model to classify music genres using preprocessed MFCC features. Includes early stopping, model checkpointing, and comprehensive evaluation.

### Key Features
- **Automatic GPU detection**: Uses CUDA if available, falls back to CPU
- **Early stopping**: Stops training when validation loss stops improving
- **Model checkpointing**: Saves best model based on validation performance
- **Comprehensive evaluation**: Provides overall and per-genre accuracy metrics
- **Regularization**: Multiple techniques to prevent overfitting (dropout, weight decay, gradient clipping)

### Usage

Simply run:

```bash
python train.py
```

The script will:
1. Check for GPU availability
2. Load preprocessed data (automatically preprocessing from JSON if needed)
3. Build and display model architecture
4. Train the model with progress updates
5. Evaluate on test set with per-genre breakdown
6. Save the trained model and metadata

### Configuration Constants

You can modify these at the top of `train.py`:

```python
MODEL_SAVE_PATH = "saved_models"           # Directory for saved models
EPOCHS = 30                                 # Maximum training epochs
BATCH_SIZE = 64                            # Batch size for training
LEARNING_RATE = 0.0008                     # Learning rate for Adam optimizer
WEIGHT_DECAY = 2e-4                        # L2 regularization strength
GRADIENT_CLIP = 1.0                        # Gradient clipping threshold
EARLY_STOPPING_PATIENCE = 8                # Epochs to wait before early stopping
```

### Model Architecture

The `GenreClassifier` model uses a CNN architecture:

```
Input (N, C, H, W)
  ↓
Conv2d(32 filters, 3x3) + ReLU
  ↓
MaxPool2d(2x2)
  ↓
BatchNorm2d + Dropout2d(0.35)
  ↓
Conv2d(48 filters, 3x3) + ReLU
  ↓
MaxPool2d(2x2)
  ↓
BatchNorm2d + Dropout2d(0.35)
  ↓
Flatten
  ↓
Linear(64) + ReLU
  ↓
BatchNorm1d + Dropout(0.7)
  ↓
Linear(num_classes)
  ↓
Output (predictions)
```

**Key Architecture Details:**
- Two convolutional blocks for feature extraction
- Batch normalization for training stability
- Multiple dropout layers (spatial and fully connected) to prevent overfitting
- Reduced model capacity (fewer filters, smaller FC layers) for better generalization

### Training Process

The training script provides detailed output:

```
============================================================
PyTorch Training Pipeline for Music Genre Classification
============================================================
✓ GPU detected: [GPU Name]  (or ⚠ GPU not available, using CPU)

Model configuration:
  Input shape: (C=X, H=Y, W=Z)
  Number of classes: 10
  Classes: ['blues', 'classical', ...]

Starting training...
Device: cuda, Batch size: 64, Learning rate: 0.0008 (fixed)
Weight decay (L2): 0.0002 (helps prevent overfitting)
Max epochs: 30, Early stopping patience: 8
------------------------------------------------------------
Epoch 1/30
  Train Loss: X.XXXX, Train Acc: XX.XX%
  Val Loss: X.XXXX, Val Acc: XX.XX%
  ✓ Saved best model (val_acc: XX.XX%)
------------------------------------------------------------
...
```

### Output Files

After training completes, the following files are saved in `saved_models/`:

1. **`best_model_pytorch.pt`**: Best model checkpoint (based on validation loss)
   - Contains model state, optimizer state, epoch number, and metrics

2. **`final_model_pytorch.pt`**: Final model state
   - Contains model state and metadata needed for inference:
     - Model architecture parameters (channels, height, width)
     - Number of classes

3. **`genre_mapping.json`**: Genre label mapping
   - Maps integer labels to genre names for inference

### Evaluation

After training, the script evaluates on the test set and prints:
- Overall test loss and accuracy
- Per-genre accuracy breakdown with counts

Example output:
```
Test Results:
  Loss: 0.XXXX
  Accuracy: XX.XX%

Per-genre accuracy:
  blues: XX.XX% (XX/XX correct)
  classical: XX.XX% (XX/XX correct)
  ...
```

---

## Workflow

### Complete Training Pipeline

1. **Prepare your data**: Ensure `data.json` exists in the project root
2. **Run training**: Execute `python train.py`
   - Preprocessing happens automatically if needed
   - Training proceeds with progress updates
   - Model is saved automatically
3. **Check results**: Review the saved models in `saved_models/` directory

### Typical First Run

```bash
# Step 1: Ensure data.json exists
# Step 2: Run training (preprocessing happens automatically)
python train.py

# Output:
# - preprocessed_data.npz (created automatically)
# - saved_models/best_model_pytorch.pt
# - saved_models/final_model_pytorch.pt
# - saved_models/genre_mapping.json
```

### Subsequent Runs

If `preprocessed_data.npz` already exists, preprocessing is skipped and data loads directly from the cached file for faster startup.

---

## Troubleshooting

### Common Issues

**GPU not detected:**
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- The script will automatically use CPU if GPU is unavailable

**Out of memory errors:**
- Reduce `BATCH_SIZE` in `train.py`
- Close other applications using GPU memory

**Preprocessing takes too long:**
- Preprocessing only happens once; subsequent runs use cached `preprocessed_data.npz`
- If you want to reprocess, delete `preprocessed_data.npz` and run again

**Poor model performance:**
- Adjust hyperparameters (learning rate, batch size, epochs)
- Check data quality and distribution
- Review the model architecture parameters

### File Requirements

- **`data.json`**: Required for first run (or if preprocessed data is deleted)
- **`preprocessed_data.npz`**: Created automatically, cached for faster loading
- **`saved_models/`**: Created automatically to store trained models

---

## Model Regularization

The training script includes several techniques to prevent overfitting:

- **L2 Regularization**: Weight decay in optimizer (2e-4)
- **Dropout**: 
  - Spatial dropout (0.35) after convolutional layers
  - Fully connected dropout (0.7) before final classification
- **Batch Normalization**: After each conv layer and FC layer
- **Gradient Clipping**: Prevents exploding gradients (clip norm = 1.0)
- **Early Stopping**: Stops when validation loss plateaus (patience = 8)
- **Reduced Model Capacity**: Smaller filters and FC layers

---

## Example Usage

### Basic Training

```bash
python train.py
```

### Force Reprocessing

```bash
# Delete cached preprocessed data
rm preprocessed_data.npz  # Linux/Mac
del preprocessed_data.npz  # Windows

# Run training (will reprocess from JSON)
python train.py
```

### Preprocess Only

```bash
python preprocess.py
```

---

## Notes

- The model uses **channels-first format** (N, C, H, W) as expected by PyTorch
- Data is automatically normalized to [0, 1] range during preprocessing
- Train/validation/test splits use stratified sampling to maintain class distribution
- Random seeds are fixed (random_state=42) for reproducibility
- The script automatically detects and uses GPU if available
- Model checkpoints include all necessary metadata for inference

---

## Support

For issues or questions:
1. Check that all dependencies are installed correctly
2. Verify `data.json` format matches expected structure
3. Review configuration constants if you need to adjust hyperparameters
4. Check console output for specific error messages

