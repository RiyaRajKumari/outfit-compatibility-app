# AI-Based Outfit Compatibility Prediction Using Deep Learning

An AI-powered application that predicts the compatibility of a fashion outfit by analyzing an image of a top and an image of a bottom using deep learning.

**Live Demo:** [Try the Outfit Compatibility App](https://ai-outfit-compatibility.streamlit.app/)  
**GitHub Repository:** [Outfit Compatibility App](https://github.com/RiyaRajKumari/outfit-compatibility-app)

## Project Overview

This project uses computer vision and deep learning to predict whether a top-bottom clothing combination is compatible. It uses a ResNet50-based Two-Tower neural network to extract visual features from clothing images and learn relationships between the two garments.

The trained model is integrated into an interactive Streamlit web application where users can upload clothing images and receive a compatibility prediction.

## Objectives

- Develop a deep-learning model for top-bottom outfit compatibility prediction.
- Extract meaningful visual features from clothing images using transfer learning.
- Learn garment relationships using feature fusion and binary classification.
- Evaluate model performance using standard classification metrics.
- Deploy the trained model through an interactive web application.

## Key Features

- Upload images of a top and a bottom.
- Extract image features using pretrained ResNet50.
- Predict outfit compatibility using a Two-Tower neural network.
- Display a compatibility score and prediction.
- Access the model through a deployed Streamlit application.

## Model Architecture

The application uses a ResNet50-based Two-Tower architecture.

1. **Image feature extraction:** ResNet50 extracts 2,048-dimensional visual features from each garment image.
2. **Feature projection:** A shared projection network maps the features to 128-dimensional embeddings.
3. **Feature fusion:** The top and bottom embeddings are combined using concatenation, absolute difference, and element-wise multiplication.
4. **Classification:** A multilayer perceptron produces a compatibility logit, which is converted into a probability using the sigmoid function.

### Architecture Flow

```text
Top Image                         Bottom Image
    ↓                                  ↓
 ResNet50                           ResNet50
    ↓                                  ↓
2048-D Features                   2048-D Features
    ↓                                  ↓
Shared Projection                Shared Projection
    ↓                                  ↓
128-D Embedding                  128-D Embedding
    └──────────────┬───────────────────┘
                   ↓
       Absolute Difference + Product
                   ↓
              Concatenation
                   ↓
           Neural Classifier
                   ↓
       Compatibility Probability
                   ↓
       Compatible / Incompatible
```

**Framework:** PyTorch  
**Backbone:** ResNet50 pretrained on ImageNet  
**Task:** Binary classification

## Dataset

The project uses the FashionVC dataset for top-bottom compatibility prediction.

- Ground-truth positive pairs are obtained from the dataset's top-bottom pair annotations.
- Negative pairs are constructed using hard-negative mining, selecting visually similar candidate bottoms while excluding known positive combinations.
- The dataset is divided into training, validation, and test sets by top ID to prevent the same top from appearing across different splits.

The evaluation dataset contains **6,236 pairs**, consisting of 3,118 positive pairs and 3,118 constructed hard-negative pairs.

*Note: The negative pairs are constructed training/evaluation examples; they are not all independently human-annotated as incompatible.*

## Image Preprocessing

- Resize images to 224 × 224 pixels.
- Convert images to tensors.
- Normalize using ImageNet mean and standard deviation.
- Apply random horizontal flipping during training.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Backbone | ResNet50 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Maximum epochs | 15 |
| Loss function | Binary Cross-Entropy with Logits |
| Checkpoint selection | Validation loss |

## Model Evaluation

The final reported Experiment 6 results on the held-out test set are:

| Metric | Score |
|---|---:|
| Accuracy | 96.41% |
| Precision | 96.29% |
| Recall | 96.54% |
| F1-score | 96.41% |
| ROC-AUC | 0.9931 |

These results apply to the constructed FashionVC test pairs. They should not be interpreted as a guarantee of the same performance on arbitrary real-world clothing images.

## Technology Stack

- Python
- PyTorch
- Torchvision
- ResNet50 and transfer learning
- NumPy
- Pillow
- Streamlit
- Git and Git LFS

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RiyaRajKumari/outfit-compatibility-app.git
cd outfit-compatibility-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

**Important:** The repository uses Git LFS for model files. Make sure Git LFS is installed and the model files have been downloaded correctly before running the application.

## Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** https://ai-outfit-compatibility.streamlit.app/

Model checkpoints are managed using Git Large File Storage (Git LFS).

## Limitations

- Compatibility predictions depend on the patterns learned from the FashionVC dataset.
- The model may perform differently on images from sources or clothing categories not represented in the training data.
- A high predicted probability is not a calibrated guarantee that an outfit is objectively fashionable.
- The model evaluates top-bottom compatibility and does not assess an entire outfit or individual style preferences.

## Future Improvements

- Evaluate generalization on additional fashion datasets.
- Calibrate prediction probabilities.
- Extend the model to additional garment categories.
- Compare alternative feature-fusion strategies and model architectures.
- Improve robustness to different image backgrounds, poses, and lighting conditions.

## Author

**Riya Raj Kumari**

Developed as a deep-learning project exploring computer vision, transfer learning, outfit compatibility prediction, model evaluation, and web application deployment.
