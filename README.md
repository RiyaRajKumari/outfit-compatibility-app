# AI Outfit Compatibility — Streamlit Deployment

Deployment of Experiment 6: **ResNet50 + Two-Tower compatibility model**.

## Files

```text
app.py
requirements.txt
resnet50-11ad3fa6.pth
resnet50_two_tower_epoch6.pt
```

Copy the two model files from the Kaggle project into this folder.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Experiment 6 benchmark

Accuracy 96.52% • Precision 97.98% • Recall 95.00% • F1 96.47% • ROC-AUC 0.9930.

The benchmark used constructed hard negatives, so these figures should not be presented as human-annotated real-world compatibility accuracy.
