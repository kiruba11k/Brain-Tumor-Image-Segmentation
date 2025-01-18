# Brain Tumor Image Analysis and Semantic Segmentation

This project focuses on the analysis and semantic segmentation of brain tumor images using a custom deep learning pipeline implemented in PyTorch. It processes COCO-format annotations, builds a custom dataset class, and trains a neural network model for precise tumor detection.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation Guide](#installation-guide)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Future Enhancements](#future-enhancements)
9. [Acknowledgements](#acknowledgements)

---

## Project Overview

The primary goal of this project is to automate the identification and localization of brain tumors using medical imaging data. Using bounding box annotations and a deep learning model, the project offers a robust framework for detecting tumors. This is particularly valuable for aiding medical professionals in diagnostics and research.

---

## Features

- **Data Processing**: Handles loading, preprocessing, and splitting of brain tumor datasets.
- **COCO Annotation Support**: Reads and processes bounding boxes and categories from JSON files.
- **Custom Dataset Class**: Implements a flexible PyTorch dataset class (`BrainDataset`) for easy data handling.
- **Visualization**: Provides functions to visualize images, annotations, and predictions.
- **Training Pipeline**: Defines data loaders, transformations, and training loops for semantic segmentation.
- **Model Evaluation**: Evaluates model accuracy and performance on unseen test data.

---

## Dataset

The dataset used in this project is a **brain tumor image dataset** with COCO-format annotations, including bounding boxes and labels. The dataset structure is:
```
Dataset_brain/
├── train/
├── valid/
├── test/
└── annotations/
```

You can download the dataset from [Kaggle](https://www.kaggle.com/pkdarabi/brain-tumor-image-dataset-semantic-segmentation).

---

## Installation Guide

### Prerequisites
- Python 3.7+
- Libraries: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `PIL`

### Steps to Set Up the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kiruba11k/Brain-Tumor-Image-Segmentation.git
   cd Brain-Tumor-Image-Segmentation
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and Prepare the Dataset**
   - Download the dataset from Kaggle.
   - Extract it to a folder named `Dataset_brain` in the project directory.

5. **Verify Dataset Structure**
   Ensure the dataset is organized as shown above.

---

## Usage

### 1. Preprocess and Visualize the Data
- Load the images and COCO annotations.
- Visualize samples with bounding boxes using the provided functions.

### 2. Train the Model
- Configure hyperparameters like `batch_size`, `learning_rate`, and `num_epochs` in the code.
- Execute the training cells in the Jupyter notebook.

### 3. Evaluate the Model
- Test the trained model on the validation and test datasets.
- Visualize the predicted bounding boxes on test images.

### 4. Extend the Model
- Implement additional metrics or loss functions to improve performance.
- Adapt the dataset and pipeline to other medical imaging tasks.

---

## Project Structure

```
brain-tumor-segmentation/
├── Dataset_brain/               # Dataset directory
├── ProjectBrainTumor.ipynb      # Jupyter notebook with code
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Results

- **Training Performance**: The model successfully learns from the training set and produces bounding box predictions.
- **Validation Results**: Provides quantitative metrics for accuracy and bounding box overlap.
- **Visualizations**: Displays ground truth and predicted bounding boxes for test images.

---

## Future Enhancements

- **Model Optimization**: Use pre-trained models (e.g., ResNet, EfficientNet) for better accuracy.
- **Multi-Class Segmentation**: Extend the model to detect different types of brain abnormalities.
- **Integration with Clinical Data**: Combine imaging with patient data for more robust predictions.
- **Deployment**: Create a web or mobile application for real-time tumor detection.

---

## Acknowledgements

This project is built on the **Brain Tumor Image Dataset** from [Kaggle](https://www.kaggle.com/pkdarabi/brain-tumor-image-dataset-semantic-segmentation). The implementation uses **PyTorch** for deep learning and **COCO annotations** for dataset management.

---
