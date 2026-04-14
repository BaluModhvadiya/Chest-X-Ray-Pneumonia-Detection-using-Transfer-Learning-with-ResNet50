# Chest X-Ray Pneumonia Detection using Transfer Learning with ResNet50

This project uses transfer learning with ResNet50 to classify chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

The model is built with TensorFlow/Keras and uses image augmentation, validation splitting, and evaluation metrics such as classification report and confusion matrix.

## Project Structure

```bash
chest-xray-pneumonia-detection/
│
├── data/
│   └── chest_xray/
│       └── train/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── notebooks/
│   └── pneumonia_detection_resnet50.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   ├── plots/
│   └── models/
│
├── .gitignore
├── requirements.txt
└── README.md
