# 👗 AI Outfit Compatibility Prediction Using Deep Learning

A deep-learning application that predicts whether a top and a bottom clothing item are visually compatible. The project combines **ResNet50 feature extraction**, a **Two-Tower neural network**, and **hard-negative mining** to learn relationships between clothing items.

🔗 **Live Demo:** [AI Outfit Compatibility](https://ai-outfit-compatibility.streamlit.app/)

## 📌 Project Overview

Choosing clothing items that work well together is an important part of fashion recommendation. This project explores how deep learning can automatically evaluate the visual compatibility of a top-bottom clothing combination.

The system accepts two clothing images and predicts whether the pair is compatible or incompatible. A trained model is integrated into an interactive Streamlit web application.

## 🎯 Objectives

- Extract meaningful visual features from clothing images.
- Learn relationships between tops and bottoms.
- Improve classification using hard-negative mining.
- Compare multiple model architectures and training strategies.
- Evaluate performance using classification metrics.
- Deploy the trained model as a web application.

## 🗂️ Dataset

The project uses the **FashionVC dataset**.

| Dataset characteristic | Value |
|---|---:|
| Positive top-bottom pairs | 20,715 |
| Unique tops | 14,866 |
| Unique bottoms | 13,660 |
| Final test pairs | 6,236 |

Positive pairs are obtained from the dataset's ground-truth combinations. Negative pairs are constructed by pairing tops with incorrect bottoms. The final experiment uses visually similar hard negatives to make classification more challenging.

**Important:** The negative pairs are constructed examples, not independently human-annotated incompatible outfits.

### Data Preparation

- Images resized to `224 × 224` pixels.
- Images converted to PyTorch tensors.
- ImageNet mean and standard deviation normalization applied.
- Random horizontal flipping used during training.
- Data split at the top-item level into approximately 70% training, 15% validation, and 15% testing.

The split prevents the same top item from appearing across multiple splits. Bottom items were not independently isolated across the splits.

## 🧠 Model Architecture

The final model combines **ResNet50** with a Two-Tower compatibility network.

1. **Feature extraction:** ResNet50 extracts a 2,048-dimensional feature vector from each clothing image.
2. **Shared projection:** Each feature vector is projected into a 128-dimensional embedding space through a shared projection network.
3. **Relationship modelling:** The model calculates the absolute difference and element-wise product of the two embeddings.
4. **Classification:** The combined relationship features are passed to a neural classifier.
5. **Prediction:** The model outputs a compatibility probability and a binary prediction.

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

## 🧪 Experimental Progress

Six experiments were conducted to explore different architectures and training strategies.

| Experiment | Approach | Reported result |
|---|---|---:|
| 1 | Frozen ResNet18 + MLP | 60.83% validation accuracy |
| 2 | Relationship-aware features | 60.93% validation accuracy |
| 3 | Fine-tuned ResNet18 | 56.50% selected validation accuracy |
| 4 | ResNet18 Two-Tower | 61.57% selected validation accuracy |
| 5 | Two-Tower + hard-negative mining | 62.79% validation accuracy |
| 6 | ResNet50 + Two-Tower + hard-negative mining | **96.41% test accuracy** |

These are the reported results from the respective experiments. Experiment 6 used a different backbone and experimental setup, so the results should not be interpreted as a perfectly controlled comparison of backbone architectures alone.

## ⚙️ Training Configuration

| Parameter | Configuration |
|---|---|
| Backbone | ResNet50, pretrained on ImageNet |
| Input image size | 224 × 224 |
| Feature dimension | 2,048 |
| Embedding dimension | 128 |
| Loss function | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Training epochs | 15 |
| Selected checkpoint | Epoch 8, based on validation loss |

The selected checkpoint achieved a validation loss of **0.1094** and validation accuracy of **96.67%**.

## 📊 Final Test Results — Experiment 6

The selected model was evaluated on 6,236 held-out pairs.

| Metric | Result |
|---|---:|
| Accuracy | **96.41%** |
| Precision | 96.29% |
| Recall | 96.54% |
| F1-score | 96.41% |
| ROC-AUC | **0.9931** |

### Confusion Matrix

| Actual class | Predicted incompatible | Predicted compatible |
|---|---:|---:|
| Incompatible | 3,002 | 116 |
| Compatible | 108 | 3,010 |

The model correctly classified 6,012 out of 6,236 test pairs.

**Evaluation note:** These metrics measure performance on the constructed FashionVC hard-negative benchmark. They do not establish equivalent accuracy on arbitrary real-world outfits.

## 🚀 Streamlit Application

The deployed application allows users to:

- Upload a top image.
- Upload a bottom image.
- Generate a compatibility prediction.
- View the model's compatibility score.
- Inspect technical details and benchmark metrics.

**Try the application:** [AI Outfit Compatibility — Live Demo](https://ai-outfit-compatibility.streamlit.app/)

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet50
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Kaggle
- Git and GitHub

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RiyaRajKumari/outfit-compatibility-app.git
cd outfit-compatibility-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the model files

Ensure the following files are present in the project directory:

- `resnet50-11ad3fa6.pth`
- `resnet50_two_tower_epoch6.pt`

The pretrained ResNet50 weights and trained compatibility checkpoint are required for inference.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## ⚠️ Limitations

- Negative examples are constructed rather than independently human-annotated.
- The test split separates top items, but not necessarily bottom items.
- Predictions on arbitrary internet images may be unreliable because their visual distribution can differ from FashionVC.
- The compatibility score is a model output, not a calibrated measure of human fashion preference.
- The current system evaluates a top-bottom pair; it does not yet provide personalized recommendations.

## 🔮 Future Improvements

- Evaluate using human-annotated incompatible outfit pairs.
- Independently separate both top and bottom items across data splits.
- Improve generalization to images from different sources.
- Add personalized fashion recommendations and candidate ranking.
- Extend compatibility prediction to additional clothing categories.
- Explore scalable inference using precomputed embeddings and an API.

## 👩‍💻 Project Author

**Riya Raj Kumari**

Developed as a deep-learning project exploring visual feature extraction, relationship-aware learning, hard-negative mining, model evaluation, and application deployment.

---

*This project is an experimental fashion compatibility system intended for learning and research. Its benchmark results should not be interpreted as a guarantee of real-world outfit compatibility.*
