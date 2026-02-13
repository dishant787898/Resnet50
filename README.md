
---

# Automated Waste Classification using Fine-Tuned ResNet50

An industrial-grade Computer Vision project designed to automate the classification of waste into **Organic** and **Recyclable** categories. This system leverages transfer learning with the **ResNet50** architecture to achieve high accuracy and robust generalization on large-scale environmental datasets.

## 🚀 Features

* **Advanced Transfer Learning**: Utilizes the pre-trained **ResNet50** weights (ImageNet) as a feature extractor, followed by a custom-designed classification head for binary waste sorting.
* **Large-Scale Data Handling**: Trained and validated on a massive dataset of **22,000+ images**, ensuring the model can handle a wide variety of waste object orientations and lighting conditions.
* **Performance Optimization**: Implements advanced training callbacks including `ReduceLROnPlateau` for adaptive learning rate adjustments and `EarlyStopping` to prevent overfitting.
* **Evaluation Metrics**: Includes detailed performance analysis through **Confusion Matrices**, **Accuracy/Loss Evolution curves**, and classification reports.
* **Production Ready**: The final model is exported in `.h5` format, making it compatible with web-based inference engines like Streamlit or IoT-enabled smart bins.

## 🛠️ Tech Stack

* **Deep Learning Framework**: TensorFlow / Keras.
* **Computer Vision**: OpenCV.
* **Architecture**: ResNet50.
* **Data Science**: NumPy, Pandas, Scikit-learn.
* **Visualization**: Matplotlib, Seaborn.

## 🏗️ Model Architecture

The model follows a tiered deep learning architecture:

1. **Base Layer**: ResNet50 convolutional base (frozen or fine-tuned).
2. **Global Average Pooling**: Reduces spatial dimensions to a 1D vector.
3. **Dropout Layer (0.5)**: Strategic regularization to improve the model's ability to generalize to new, unseen waste images.
4. **Dense Classifier**: A fully connected layer with a **Sigmoid** activation for high-precision binary classification (Organic vs. Recyclable).

## 📈 Performance Results

* **Training Accuracy**: Stabilizes around **96%**.
* **Validation Accuracy**: Reaches nearly **98%**, indicating strong generalization and minimal overfitting.
* **Loss Evolution**: Consistent downward movement of both training and validation loss curves, reflecting robust model convergence.

## 💻 How to Use

### 1. Installation

```bash
pip install tensorflow opencv-python matplotlib pandas scikit-learn

```

### 2. Dataset Setup

The system is designed to work with the **Waste-Classification-Data** dataset (available on Kaggle). Ensure the directory structure follows:

```text
/DATASET
    /TRAIN
        /O (Organic)
        /R (Recyclable)
    /TEST

```

### 3. Execution

Open the `ResNet50.ipynb` notebook in Google Colab or Jupyter and run the cells sequentially to mount Google Drive, preprocess data, and begin training.

---
