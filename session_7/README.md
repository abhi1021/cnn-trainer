# Session 7: Advanced CNN Architecture for CIFAR-10

## Overview
This session implements an advanced Convolutional Neural Network (CNN) for CIFAR-10 image classification, incorporating modern architectural techniques including dilated convolutions, depthwise separable convolutions, and strategic use of pooling layers.

## Model Architecture

### Network Structure
The model follows a systematic progression through 4 main convolution blocks (C1-C4) plus an output block:

```
Input: 3×32×32 RGB Image
│
├── C1: Convolution Block 1 (RF: 3→9)
│   ├── Conv2d(3→32, k=3, p=0)      # 32×32 → 30×30, RF=3
│   ├── Conv2d(32→64, k=3, p=0)     # 30×30 → 28×28, RF=5
│   └── Conv2d(64→64, k=3, d=2)     # 28×28 → 24×24, RF=9 (Dilated)
│
├── C2: Convolution Block 2 (RF: 9→13)
│   ├── Conv2d(64→32, k=1)          # Channel reduction
│   ├── DepthwiseConv(32, k=3, p=1) # 24×24, RF=11
│   ├── PointwiseConv(32→64, k=1)   # 24×24, RF=11
│   └── Conv2d(64→64, k=3, p=0)     # 24×24 → 22×22, RF=13
│
├── C3: Convolution Block 3 (RF: 13→21)
│   ├── Conv2d(64→64, k=3, s=2)     # 22×22 → 10×10, RF=17 (Strided)
│   ├── Conv2d(64→32, k=1)          # Channel reduction
│   └── AvgPool2d(2×2)              # 10×10 → 5×5, RF=21
│
├── C4: Convolution Block 4 (RF: 21→44)
│   ├── Conv2d(32→64, k=3, p=1)     # 5×5, RF=25
│   ├── Conv2d(64→64, k=3, p=1)     # 5×5, RF=29
│   └── AvgPool2d(5×5)              # 5×5 → 1×1, RF=44 (GAP)
│
└── Output: Conv2d(64→10, k=1)      # 1×1×10 logits
    └── LogSoftmax
```

### Key Architectural Features

1. **Dilated Convolutions (C1)**
   - Used in the first block with dilation=2
   - Increases receptive field without reducing spatial dimensions
   - Captures broader context early in the network

2. **Depthwise Separable Convolutions (C2)**
   - Separates spatial and channel-wise operations
   - Reduces parameters while maintaining expressiveness
   - Depthwise: 32 groups for 32 channels (288 params)
   - Pointwise: 1×1 conv for channel mixing (2,048 params)

3. **Strategic Pooling**
   - Strided convolution (C3): Combines feature extraction with downsampling
   - Average pooling: Used for smooth dimension reduction
   - Global Average Pooling (GAP): Reduces spatial dimensions to 1×1

4. **Channel Management**
   - 1×1 convolutions for efficient channel reduction
   - Progressive channel expansion: 3 → 32 → 64
   - Strategic reduction before pooling to manage parameters

## Receptive Field Analysis

### Calculation Formula
```
RF_out = RF_in + (kernel_size - 1) × jump_in
jump_out = jump_in × stride
```

### Layer-by-Layer Breakdown

| Layer | Operation | Input Size | Output Size | Kernel | Stride | Dilation | RF | Jump |
|-------|-----------|------------|-------------|--------|--------|----------|----|----|
| Input | - | 32×32×3 | 32×32×3 | - | - | - | 1 | 1 |
| C1-1 | Conv2d | 32×32×3 | 30×30×32 | 3 | 1 | 1 | 3 | 1 |
| C1-2 | Conv2d | 30×30×32 | 28×28×64 | 3 | 1 | 1 | 5 | 1 |
| C1-3 | Conv2d (d=2) | 28×28×64 | 24×24×64 | 3 | 1 | 2 | 9 | 1 |
| C2-1 | Conv2d (1×1) | 24×24×64 | 24×24×32 | 1 | 1 | 1 | 9 | 1 |
| C2-2 | Depthwise | 24×24×32 | 24×24×32 | 3 | 1 | 1 | 11 | 1 |
| C2-3 | Pointwise | 24×24×32 | 24×24×64 | 1 | 1 | 1 | 11 | 1 |
| C2-4 | Conv2d | 24×24×64 | 22×22×64 | 3 | 1 | 1 | 13 | 1 |
| C3-1 | Conv2d (s=2) | 22×22×64 | 10×10×64 | 3 | 2 | 1 | 17 | 2 |
| C3-2 | Conv2d (1×1) | 10×10×64 | 10×10×32 | 1 | 1 | 1 | 17 | 2 |
| C3-3 | AvgPool | 10×10×32 | 5×5×32 | 2 | 2 | 1 | 21 | 4 |
| C4-1 | Conv2d | 5×5×32 | 5×5×64 | 3 | 1 | 1 | 25 | 4 |
| C4-2 | Conv2d | 5×5×64 | 5×5×64 | 3 | 1 | 1 | 29 | 4 |
| C4-3 | GAP | 5×5×64 | 1×1×64 | 5 | 1 | 1 | **44** | 4 |
| Output | Conv2d (1×1) | 1×1×64 | 1×1×10 | 1 | 1 | 1 | 44 | 4 |

**Final Receptive Field: 44×44** - Covers full 32×32 input with margin

## Model Parameters

### Total Parameters: **193,280**

#### Parameter Distribution by Block

| Block | Operation | Parameters | % of Total |
|-------|-----------|------------|-----------|
| **C1** | | **56,352** | **29.2%** |
| | Conv2d (3→32) | 864 | 0.4% |
| | BatchNorm2d (32) | 64 | 0.03% |
| | Conv2d (32→64) | 18,432 | 9.5% |
| | BatchNorm2d (64) | 128 | 0.07% |
| | Conv2d dilated (64→64) | 36,864 | 19.1% |
| | BatchNorm2d (64) | 128 | 0.07% |
| **C2** | | **41,472** | **21.5%** |
| | Conv2d 1×1 (64→32) | 2,048 | 1.1% |
| | Depthwise (32) | 288 | 0.1% |
| | BatchNorm2d (32) | 64 | 0.03% |
| | Pointwise (32→64) | 2,048 | 1.1% |
| | BatchNorm2d (64) | 128 | 0.07% |
| | Conv2d (64→64) | 36,864 | 19.1% |
| | BatchNorm2d (64) | 128 | 0.07% |
| **C3** | | **39,296** | **20.3%** |
| | Conv2d strided (64→64) | 36,864 | 19.1% |
| | BatchNorm2d (64) | 128 | 0.07% |
| | Conv2d 1×1 (64→32) | 2,048 | 1.1% |
| **C4** | | **55,520** | **28.7%** |
| | Conv2d (32→64) | 18,432 | 9.5% |
| | BatchNorm2d (64) | 128 | 0.07% |
| | Conv2d (64→64) | 36,864 | 19.1% |
| | BatchNorm2d (64) | 128 | 0.07% |
| **Output** | | **640** | **0.3%** |
| | Conv2d 1×1 (64→10) | 640 | 0.3% |

### Memory Footprint
- **Input Size:** 0.01 MB (3×32×32)
- **Forward/Backward Pass:** 6.49 MB
- **Parameters:** 0.74 MB
- **Estimated Total:** 7.24 MB

## Training Configuration

### Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Hyperparameters
- **Optimizer:** SGD
  - Learning Rate: 0.01 (base)
  - Momentum: 0.9
  - Weight Decay: 1e-4
- **Scheduler:** OneCycleLR
  - Max LR: 0.05
  - Epochs: 30
  - Steps per epoch: 391
- **Batch Size:** 128
- **Dropout:** 0.05

## Training Results

### Performance Summary
- **Final Test Accuracy:** 87.01%
- **Final Test Loss:** 0.3850
- **Training Epochs:** 30
- **Best Accuracy:** 87.01% (Epoch 30)

### Epoch-by-Epoch Progress

| Epoch | Train Acc | Test Acc | Test Loss | Notes |
|-------|-----------|----------|-----------|-------|
| 1 | 41.61% | 55.07% | 1.2453 | Initial training |
| 5 | 70.21% | 72.40% | 0.7894 | Rapid improvement |
| 10 | 78.26% | 79.27% | 0.5944 | Approaching 80% |
| 15 | 81.51% | 81.65% | 0.5274 | Steady progress |
| 20 | 84.31% | 83.98% | 0.4732 | Breaking 84% |
| 25 | 87.29% | 86.08% | 0.4067 | Strong performance |
| 30 | 89.55% | **87.01%** | **0.3850** | Final model |

### Training Characteristics
- **Convergence:** Smooth and consistent
- **Overfitting:** Minimal (2.54% gap at epoch 30)
- **Learning Rate Schedule:** OneCycleLR effectively utilized
- **Generalization:** Good test performance maintained

## Key Observations

### Architectural Benefits
1. **Efficient Parameter Usage:** ~193K parameters achieve 87% accuracy
2. **Adequate Receptive Field:** RF=44 covers 32×32 input completely
3. **Modern Techniques:** Dilated + Depthwise convolutions reduce parameters
4. **Progressive Learning:** Gradual feature abstraction through blocks

### Training Insights
1. **OneCycleLR Effectiveness:** Smooth convergence without oscillations
2. **Regularization Balance:** Dropout=0.05 prevents overfitting
3. **Data Augmentation:** Random rotation and translation help generalization
4. **Batch Normalization:** Stabilizes training across all blocks

## Usage

### Training
```python
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Setup optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=0.05, 
                       steps_per_epoch=len(train_loader), epochs=30)

# Training loop
for epoch in range(1, 31):
    train(model, device, train_loader, optimizer, scheduler, epoch)
    test(model, device, test_loader)
```

### Inference
```python
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1)
```

## Files
- `session_7.py` - Model definition and architecture
- `session_7.ipynb` - Training notebook with experiments and results
- `README.md` - This documentation file

## Requirements
```
torch>=1.8.0
torchvision>=0.9.0
torchsummary
tqdm
numpy
matplotlib
```

## Conclusion
This architecture demonstrates effective use of modern CNN techniques to achieve 87% accuracy on CIFAR-10 with only 193K parameters. The combination of dilated convolutions, depthwise separable convolutions, and strategic pooling creates an efficient yet powerful model suitable for resource-constrained environments.
