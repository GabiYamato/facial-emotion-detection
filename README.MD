# 🎭 Facial Emotion Detection Deep Learning Project

A production-ready PyTorch deep learning repository for **Facial Emotion Recognition** on 15,339 aligned face images across 7 emotion categories (*Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral*).

---

## 📁 Repository Structure

```
facial emotion detection/
├── data/
│   ├── DumpStack.log               # Dataset ZIP archive (15,339 images)
│   └── extracted/                  # Extracted dataset into folders 1..7
├── data_analysis/
│   ├── eda_facial_emotion_detection.ipynb  # Interactive EDA Notebook
│   ├── emotion_class_distribution.png       # Class distribution plot
│   └── emotion_sample_grid.png              # 7x5 Face samples per emotion
├── docs/
│   ├── dataset_guide.md            # Dataset specification & class breakdown
│   └── architecture.md             # Deep learning architecture & workflow
├── saved_models/                   # Model weights & PyTorch checkpoints
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Paths, label mappings, hyperparameters
│   ├── dataset.py                  # PyTorch Dataset & DataLoader
│   ├── models.py                   # Custom EmotionCNN & ResNet-18 architectures
│   ├── utils.py                    # Checkpointing & metric visualization
│   ├── train.py                    # Training & validation loop
│   └── evaluate.py                 # Test evaluation & confusion matrix
└── README.MD                       # Project documentation
```

---

## ⚙️ Virtual Environment Setup

The virtual environment is created in the parent folder:
`d:\Deep learning\venv`

### Activate Environment (Windows PowerShell):
```powershell
& "d:\Deep learning\venv\Scripts\Activate.ps1"
```

---

## 📊 Exploratory Data Analysis (EDA)

The Exploratory Data Analysis Jupyter Notebook is available at:
`data_analysis/eda_facial_emotion_detection.ipynb`

### Key EDA Highlights:
- **Total Images**: 15,339 aligned facial cropped images.
- **Emotion Classes**:
  - `1`: Surprise (1,619 images)
  - `2`: Fear (355 images)
  - `3`: Disgust (877 images)
  - `4`: Happiness (5,957 images)
  - `5`: Sadness (2,460 images)
  - `6`: Anger (867 images)
  - `7`: Neutral (3,204 images)
- **Class Imbalance**: Happiness represents 38.8% of the dataset, whereas Fear (2.3%) and Anger (5.7%) are minority classes. Class weighting and data augmentation are implemented in `src/` to handle this.

---

## 🚀 Quick Start & Usage

### 1. Run EDA Notebook
Open Jupyter Notebook from the virtual environment:
```powershell
& "d:\Deep learning\venv\Scripts\jupyter.exe" notebook data_analysis/eda_facial_emotion_detection.ipynb
```

### 2. Train Model
Train the default custom `EmotionCNN` or `ResNet18` model:
```powershell
& "d:\Deep learning\venv\Scripts\python.exe" -m src.train
```

### 3. Evaluate Model
Evaluate the trained checkpoint on test set:
```powershell
& "d:\Deep learning\venv\Scripts\python.exe" -m src.evaluate
```

---

## 🛠️ Requirements & Tech Stack
- **Language**: Python 3.12+
- **Deep Learning Framework**: PyTorch & Torchvision
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib, Seaborn, OpenCV, Pillow
- **Machine Learning Utilities**: Scikit-Learn, Albumentations, Tqdm
