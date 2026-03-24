# DeepLearning Project

Detection of AI-generated images using fine-tuned CNNs with interpretability techniques.

---

## Overview

This repository contains two implementations of classification models using different loss functions:

- **Binary Cross Entropy (BCE)**
- **Cross Entropy (CE)**

Both models are built using **TensorFlow/Keras** and demonstrate the differences and practical applications of these loss functions in image classification tasks.

The project focuses on **detecting AI-generated images** using fine-tuned Convolutional Neural Networks (CNNs), with an emphasis on **model interpretability** through **Grad-CAM**. Performance is benchmarked using the **CIFAKE dataset**.

---

## Repository Structure
Binary_cross_entropy_model.ipynb: A notebook implementing a binary classification model optimized with Binary Cross Entropy loss.
Cross_entropy_model.ipynb: A notebook implementing a classification model optimized with Cross Entropy loss.

## Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Jupyter Notebook or JupyterLab
- TensorFlow ≥ 2.6
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib
```


## Usage
1. Clone this repository:
```bash
git clone https://github.com/frmarker/DeepLearningProject.git
cd DeepLearningProject
```

2. Open the notebooks using Jupyter:
```bash
jupyter notebook
```

3. Run each cell sequentially to train and evaluate the models.
   

## Model Details

### 🔹 Binary Cross Entropy Model

**File:** `Binary_cross_entropy_model.ipynb`

- Uses **Binary Cross Entropy (BCE)** loss  
- Suitable for **binary classification**  
- Single output neuron with **sigmoid activation**

**Key components:**

- Data preprocessing for binary labels  
- CNN architecture:
  - Input layer  
  - Hidden layers (ReLU activation)  
  - Output layer (Sigmoid)  
- Metrics: Accuracy and loss  

---

### 🔹 Cross Entropy Model

**File:** `Cross_entropy_model.ipynb`

- Uses **Cross Entropy (CE)** loss  
- Typically used for **multi-class classification**  
- Output layer with **softmax activation**

**Key components:**

- Data preprocessing for class labels  
- CNN architecture:
  - Input layer  
  - Hidden layers (ReLU activation)  
  - Output layer (Softmax)  
- Metrics: Accuracy and loss  

---

## Results

Both models include visualizations of:

- Training vs. validation accuracy  
- Training vs. validation loss  

### Performance

| Model | Test Accuracy |
|------|--------------|
| Cross Entropy | **95.2%** |
| Binary Cross Entropy | **94.7%** |

---

## Interpretability

To enhance model transparency, **Grad-CAM** was used to visualize:

- Image regions influencing predictions  
- Model attention patterns  

This helps validate whether the model focuses on meaningful features when distinguishing real vs. AI-generated images.

---

## Notes

- The CIFAKE dataset is used for benchmarking  
- The project demonstrates both **performance comparison** and **interpretability analysis**
