# MNIST Digit Classifier (PyTorch)

This project trains a **Convolutional Neural Network (CNN)** on the
MNIST dataset (handwritten digits 0--9).\
The goal is to keep the model **small (\~25K parameters)** while still
achieving **95%+ accuracy**.

------------------------------------------------------------------------

## 📂 Project Structure

-   **Code Blocks 2--5** → Setup (device, transforms, dataset,
    dataloaders)\
-   **Code Block 6** → Show a sample batch of images\
-   **Code Block 7** → CNN model (`Net`)\
-   **Code Block 8** → Placeholders for training/test results\
-   **Code Block 9** → Training and testing functions\
-   **Code Block 10** → Main training loop with optimizer, scheduler,
    and early stopping

------------------------------------------------------------------------

## 🧠 Model Architecture

The network is very lightweight but powerful enough for MNIST:

    Input (1×28×28 grayscale)
     ↓ Conv(16 filters, 3×3, padding=1) + BatchNorm + ReLU
     ↓ MaxPool(2×2)   → 16×14×14
     ↓ Conv(32 filters, 3×3, padding=1) + BatchNorm + ReLU
     ↓ MaxPool(2×2)   → 32×7×7
     ↓ Flatten        → 1568 features
     ↓ Fully Connected (1568 → 64 or 128)
     ↓ Dropout (0.25)
     ↓ Fully Connected (→ 10 classes)

-   Total parameters: \~25K\
-   Small enough for fast training\
-   Still reaches **95--98% accuracy**

------------------------------------------------------------------------

## ⚙️ Training Setup

-   **Optimizer**: Adam (`lr=0.001`)\
-   **Loss**: CrossEntropyLoss (standard for classification)\
-   **Scheduler**: ReduceLROnPlateau (reduces LR if accuracy plateaus)\
-   **Batch Size**: 128\
-   **Transforms**: Just normalization (no heavy augmentations, since
    MNIST is already clean)

We also use **early stopping** --- training ends automatically once test
accuracy passes 95%.

------------------------------------------------------------------------

## 📊 Results

-   **Epoch 1** → \~94--95% accuracy\
-   **Epoch 2--3** → \~96--97% accuracy\
-   **Epoch 5** → \~98% accuracy (with fc1 = 128)

This is impressive for such a tiny network.

------------------------------------------------------------------------

## 🚀 How to Run

1.  Install dependencies:

    ``` bash
    pip install torch torchvision matplotlib tqdm
    ```

2.  Run the training script (Python file with all code blocks).\

3.  Training stops once accuracy \>95% (usually 1--2 epochs).

------------------------------------------------------------------------

## 🔑 Key Learnings

-   **Batch size matters**: Smaller batches (128 vs 1000) = faster
    convergence.\
-   **Data augmentation**: Random crops/rotations actually slowed down
    early accuracy.\
-   **Adam \> SGD**: Adam optimizer helped hit 95% in the first 1--2
    epochs.\
-   **MaxPooling**: Helps shrink feature maps quickly → simpler FC layer
    → better convergence.\
-   **Early stopping**: Saves time once the target accuracy is reached.

------------------------------------------------------------------------

## 📌 Next Steps

-   Try training for more epochs (up to 10) → reach 98%+\
-   Experiment with dropout values (0.25 → 0.5)\
-   Replace MaxPool with strided convolutions (like modern CNNs)\
-   Deploy this trained model on AWS Lambda or a simple Flask API

------------------------------------------------------------------------
