# CNN Trainer - MNIST Classification Models

This repository contains three iterations of CNN models for MNIST digit classification, progressively improving accuracy while exploring different architectural choices.

## Model 1 (model_1.py)

**Target:** Establish baseline CNN architecture with basic convolution blocks
**Results:**
- Parameters: 5,124
- Best Train Accuracy: 99.06%
- Best Test Accuracy: 99.13%

**Analysis:**
Good baseline model with efficient parameter usage. The model uses a 7x7 kernel in the final layer to reduce feature maps to 1x1, eliminating the need for GAP. Shows slight overfitting in later epochs as train accuracy lags behind test accuracy. The architecture effectively uses transition blocks with 1x1 convolutions for dimensionality reduction.

## Model 2 (model_2.py)

**Target:** Replace large kernel with Global Average Pooling (GAP) for better generalization
**Results:**
- Parameters: 5,588
- Best Train Accuracy: 99.14%
- Best Test Accuracy: 99.23%

**Analysis:**
Improved model with GAP implementation. Slightly higher parameter count but better test accuracy indicates improved generalization. The GAP layer replaces the 7x7 convolution, making the model more robust to spatial variations. Shows good learning progression with SGD optimizer and achieves consistent performance across epochs. The model demonstrates better regularization compared to Model 1.

## Model 3 (model_3.py)

**Target:** Achieve 99.4%+ accuracy with data augmentation and refined architecture
**Results:**
- Parameters: 7,822
- Best Train Accuracy: 98.50%
- Best Test Accuracy: 99.41%

**Analysis:**
Best performing model with data augmentation (RandomRotation ±7°). Higher parameter count but achieves target accuracy of 99.4%+. The gap between train and test accuracy indicates excellent generalization due to augmentation preventing overfitting. Uses more sophisticated architecture with multiple transition blocks and careful dropout placement. The model successfully learns robust features that generalize well to rotated digits, making it more practical for real-world applications.

**Receptive Field Calculation:**
Using RF formula: RF_out = RF_in + (kernel_size - 1) × stride_in

| Layer | Kernel | Stride | Padding | Output Size | RF | Jump |
|-------|--------|--------|---------|-------------|----|----- |
| Input | - | 1 | - | 28×28 | 1 | 1 |
| Conv1 | 3×3 | 1 | 0 | 26×26 | 3 | 1 |
| Conv2 | 3×3 | 1 | 0 | 24×24 | 5 | 1 |
| Conv3 | 1×1 | 1 | 0 | 24×24 | 5 | 1 |
| MaxPool | 2×2 | 2 | 0 | 12×12 | 6 | 2 |
| Conv4 | 3×3 | 1 | 0 | 10×10 | 10 | 2 |
| Conv5 | 3×3 | 1 | 0 | 8×8 | 14 | 2 |
| Conv6 | 3×3 | 1 | 0 | 6×6 | 18 | 2 |
| Conv7 | 3×3 | 1 | 1 | 6×6 | 22 | 2 |
| GAP | 3×3 | 3 | 0 | 2×2 | 26 | 6 |
| Conv8 | 2×2 | 1 | 0 | 1×1 | 32 | 6 |

**Final Receptive Field: 32×32**