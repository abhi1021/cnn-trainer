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

## 📈 Performance Features

### Training Monitoring
- Real-time progress tracking with tqdm
- Accuracy and loss tracking for both training and testing
- Automatic model evaluation after each epoch

### Visualization
- Training/Testing loss and accuracy plots
- Sample batch visualization with labels
- Comprehensive performance analysis

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
