import torch
import numpy as np
from train import GenreClassifier

data = np.load("preprocessed_data.npz")

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
mapping = list(data["mapping"])

X_test = X_test.permute(0, 3, 1, 2)
samples = X_test[:5]
C, H, W = samples.shape[1:]
NUM_CLASSES = len(data["mapping"])

model = GenreClassifier(C, H, W, NUM_CLASSES)
model.eval()

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

sample_means = []

print("\n=====================")
print("   UNTRAINED MODEL   ")
print("=====================\n")

for i in range(len(samples)):
    x = samples[i].unsqueeze(0)

    with torch.no_grad():
        logits = model(x)[0].numpy()
        probs = softmax(logits)

    print(f"Sample {i}:")
    print("  Probabilities:")
    for j, g in enumerate(mapping):
        print(f"    {g:12s}: {probs[j] * 100:.2f}%")
    print("-----------------------------------")