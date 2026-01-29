# CNN Training Dynamics: Gradients, Regularization & Transfer Learning

## Project Overview
This project explores the training behavior of Convolutional Neural Networks (CNNs) on a
binary image classification dataset. The focus is on understanding:

- Vanishing and Exploding Gradients
- Overfitting and Underfitting
- Regularization techniques
- Custom CNN architectures
- Transfer Learning using pretrained models

---

## Dataset
- Binary image classification dataset
- Dataset is modified where required to demonstrate gradient behavior

---

## Experiments Performed

### 1. Vanishing & Exploding Gradients
- Deep CNN with improper initialization and activation functions
- Observed gradient shrinkage and explosion
- Gradient norms plotted across layers and epochs

### 2. Regularization Techniques
- L2 (Weight Decay)
- Dropout
- Data Augmentation
- Batch Normalization

Comparison of training vs validation loss to show overfitting and underfitting.

### 3. Custom CNN Model
- Designed and trained a CNN from scratch
- Evaluated on original dataset
- Accuracy, loss curves, and confusion matrix reported

### 4. Transfer Learning
- Pretrained model used (e.g. VGG16 / ResNet50)
- Fine-tuning applied on final layers
- Performance compared with custom CNN

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
