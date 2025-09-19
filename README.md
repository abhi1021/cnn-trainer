# CNN Trainer for MNIST Classification

A PyTorch implementation of a Convolutional Neural Network for MNIST digit classification, featuring modern deep learning techniques including batch normalization, dropout, and data augmentation.

## 🚀 Model Architecture

### Network Structure
The model follows a carefully designed CNN architecture optimized for MNIST classification:

```
Input (28x28x1) 
    ↓
Conv2d(1→16, 3x3) + BatchNorm + ReLU → 26x26x16
    ↓
Conv2d(16→16, 3x3) + BatchNorm + ReLU → 24x24x16
    ↓
MaxPool2d(2x2) → 12x12x16
    ↓
Conv2d(16→16, 3x3) + BatchNorm + ReLU → 10x10x16
    ↓
Conv2d(16→32, 3x3) + BatchNorm + ReLU → 8x8x32
    ↓
MaxPool2d(2x2) → 4x4x32
    ↓
Conv2d(32→8, 1x1) → 4x4x8
    ↓
Conv2d(8→16, 3x3, pad=1) + BatchNorm + ReLU → 4x4x16
    ↓
Conv2d(16→16, 3x3) + BatchNorm + ReLU → 2x2x16
    ↓
Flatten → 64
    ↓
FC(64→40) + Dropout(0.1) + ReLU
    ↓
FC(40→10) + LogSoftmax
    ↓
Output (10 classes)
```

## 📊 Model Analysis

### Total Parameter Count Test ✅
**Approximately 16,426 parameters**

**Detailed Parameter Breakdown:**
- **Convolutional Layers:**
  - conv1 (1→16): 160 parameters
  - conv2 (16→16): 2,320 parameters  
  - conv3 (16→16): 2,320 parameters
  - conv4 (16→32): 4,640 parameters
  - conv5 (32→8): 264 parameters
  - conv6 (8→16): 1,168 parameters
  - conv7 (16→16): 2,320 parameters

- **Batch Normalization Layers:** ~224 parameters
- **Fully Connected Layers:**
  - fc1 (64→40): 2,600 parameters
  - fc2 (40→10): 410 parameters

### Use of Batch Normalization ✅
**Implementation:** The model extensively uses Batch Normalization after most convolutional layers:
- `bn1`: After conv1 (16 channels)
- `bn2`: After conv2 (16 channels)  
- `bn3`: After conv3 (16 channels)
- `bn4`: After conv4 (32 channels)
- `bn6`: After conv6 (16 channels)
- `bn7`: After conv7 (16 channels)

**Benefits:**
- Accelerates training convergence
- Provides regularization effect
- Allows higher learning rates
- Reduces internal covariate shift

### Use of Dropout ✅
**Implementation:** Dropout with probability 0.1 is applied before the final classification layer:
```python
self.drop = nn.Dropout(p=0.1)
x = self.drop(F.relu(self.fc1(x)))
```

**Benefits:**
- Prevents overfitting
- Improves model generalization
- Acts as ensemble method during training

### Use of Fully Connected Layer ✅
**Implementation:** The model uses two fully connected layers instead of Global Average Pooling:

1. **FC1:** `nn.Linear(64, 40)` - Reduces feature dimensionality
2. **FC2:** `nn.Linear(40, 10)` - Final classification layer

**Architecture Choice:**
- Uses traditional FC layers instead of GAP (Global Average Pooling)
- Provides more learnable parameters for complex decision boundaries
- Suitable for MNIST's relatively simple feature space

## 🔧 Training Configuration

### Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
])
```

### Optimization
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Scheduler:** StepLR (step_size=15, gamma=0.1)
- **Batch Size:** 64
- **Epochs:** 20

### Hardware
- **Device:** CUDA-enabled GPU training
- **Memory Optimization:** Pin memory enabled for faster data loading

## 📈 Training Results & Performance

### Model Training Logs
The model was trained for 20 epochs with excellent performance progression:

**Training Summary:**
- **Total Parameters:** 16,426 (confirmed by torchsummary)
- **Training Time:** ~23 seconds per epoch on GPU
- **Final Test Accuracy:** 99.56%

### Epoch-by-Epoch Test Accuracy Results

| Epoch | Training Accuracy | Test Accuracy | Test Loss | Key Milestone |
|-------|------------------|---------------|-----------|---------------|
| **1** | 93.04% | **97.36%** | 0.0014 | Excellent first epoch performance |
| **2** | 97.67% | **98.02%** | 0.0010 | Surpassed 98% test accuracy |
| **3** | 98.19% | **98.34%** | 0.0008 | Steady improvement |
| **4** | 98.39% | **98.48%** | 0.0007 | Consistent gains |
| **5** | 98.60% | **98.68%** | 0.0006 | Approaching 99% |
| **6** | 98.65% | **98.98%** | 0.0005 | **🎯 Broke 98.9% barrier** |
| **7** | 98.74% | **98.96%** | 0.0005 | Stable high performance |
| **8** | 98.70% | **98.96%** | 0.0005 | Maintained accuracy |
| **9** | 98.81% | **99.09%** | 0.0004 | **🚀 First 99%+ result** |
| **10** | 98.87% | **99.11%** | 0.0004 | Solid 99%+ performance |
| **11** | 98.95% | **99.06%** | 0.0005 | Minor fluctuation |
| **12** | 99.00% | **99.20%** | 0.0004 | Strong recovery |
| **13** | 99.02% | **99.20%** | 0.0004 | Plateau at high accuracy |
| **14** | 99.00% | **99.04%** | 0.0005 | Slight dip |
| **15** | 99.08% | **99.28%** | 0.0004 | New peak reached |
| **16** | 99.30% | **99.47%** | 0.0003 | **✨ Exceptional performance** |
| **17** | 99.41% | **99.52%** | 0.0002 | **🏆 Peak test accuracy** |
| **18** | 99.49% | **99.55%** | 0.0002 | Near-perfect results |
| **19** | 99.47% | **99.58%** | 0.0002 | **🎖️ Highest test accuracy** |
| **20** | 99.47% | **99.56%** | 0.0002 | Consistent final performance |

### Performance Analysis

**🔍 Key Observations:**
- **Rapid Initial Convergence:** Achieved 97.36% test accuracy in just the first epoch
- **Consistent Improvement:** Steady accuracy gains throughout training
- **Excellent Generalization:** Test accuracy consistently matched or exceeded training accuracy
- **Stable Training:** No significant overfitting observed
- **Peak Performance:** Achieved maximum test accuracy of 99.58% at epoch 19

**🎯 Training Milestones:**
- **Epoch 1:** 97.36% - Excellent baseline performance
- **Epoch 2:** 98.02% - Surpassed 98% threshold  
- **Epoch 6:** 98.98% - Nearly 99% accuracy
- **Epoch 9:** 99.09% - First 99%+ result
- **Epoch 17-19:** 99.52-99.58% - Peak performance range

### Performance Features

**Training Monitoring:**
- Real-time progress tracking with tqdm progress bars
- Batch-level accuracy and loss reporting
- Automatic model evaluation after each epoch
- Learning rate scheduling with StepLR

**Visualization:**
- Training/Testing loss and accuracy plots
- Sample batch visualization with ground truth labels
- Comprehensive performance analysis charts

## 🏗️ Key Design Decisions

1. **Efficient Architecture:** Balanced depth and width for optimal parameter usage
2. **Modern Techniques:** Batch normalization for stable training
3. **Regularization:** Strategic dropout placement to prevent overfitting
4. **Data Pipeline:** Robust augmentation for improved generalization
5. **Monitoring:** Comprehensive tracking for training insights

## 🎯 Model Highlights

- ✅ **Compact Design:** Under 17K parameters
- ✅ **Modern Architecture:** BatchNorm + Dropout combination  
- ✅ **Robust Training:** Data augmentation and learning rate scheduling
- ✅ **GPU Optimized:** CUDA support with efficient data loading
- ✅ **Comprehensive Monitoring:** Real-time metrics and visualization

## 🚀 How to Run

1. Install dependencies:
```bash
pip install torch torchvision matplotlib tqdm torchsummary
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook Session_05.ipynb
```

3. The training will run for 20 epochs with automatic progress tracking

This implementation demonstrates a well-engineered CNN that balances model complexity with performance, incorporating modern deep learning best practices for effective MNIST classification.
