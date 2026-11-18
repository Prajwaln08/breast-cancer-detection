# Breast Cancer Detection

A machine learning project for breast cancer detection, built in two steps and finished with a clinical-style review portal.

> **Research use only.** This project is not a medical device and must not be used to diagnose, rule out disease or make treatment decisions.

| Step | Approach | Data | Status |
| :--: | -------- | ---- | :----: |
| 1 | Neural network (PyTorch) | Tabular cell-nuclei measurements | ✅ Done |
| 2 | U-Net ensemble, ResNet18 and YOLO comparison | Breast ultrasound images | ✅ Done |
| Portal | Streamlit app using the U-Net ensemble | Uploaded or example scans | ✅ Done |

---

## Step 1: Tabular Data with a Neural Network

### Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset, loaded directly from scikit-learn.

| Property | Value |
| -------- | ----- |
| Samples | 569 |
| Features | 30 numeric features |
| Classes | Malignant (`0`), Benign (`1`) |
| Class split | 212 malignant, 357 benign |
| Missing values | None |

The features describe the cell nuclei in a digitised image of a fine needle aspirate (FNA) of a breast mass, such as radius, texture, perimeter, area and smoothness.

### Workflow

1. **Data collection**: load the dataset and convert it to a pandas DataFrame.
2. **Data exploration**: check shape, data types, missing values and summary statistics.
3. **Exploratory Data Analysis (EDA)**: class balance, feature distributions, box plots by class and a correlation heatmap.
4. **Preprocessing**: split features and target, make a stratified 80/20 train/test split and scale the features with `StandardScaler`.
5. **Neural network**: build and train a simple network in PyTorch.
6. **Evaluation**: plot the loss curves and measure accuracy.

### Model

```
Input (30 features)  →  Hidden layer (16 neurons, ReLU)  →  Output (1 neuron)
```

| Setting | Value |
| ------- | ----- |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | Adam (learning rate 0.01) |
| Epochs | 100 |

### Results

| Dataset | Accuracy |
| ------- | :------: |
| Train | 99.1% |
| Test | 98.2% |

---

## Step 2: Breast Ultrasound Images

### Dataset

The **[BUSI dataset](https://doi.org/10.1016/j.dib.2019.104863)** (Breast Ultrasound Images, Al-Dhabyani et al., *Data in Brief*, 2020) contains normal, benign and malignant scans with hand-drawn lesion masks.

| Step | Images |
| ---- | -----: |
| Raw images | 780 |
| Duplicates, conflicting labels and test look-alikes removed | 111 |
| **Images used** | **669** |
| Cross-validation (5 grouped folds) | 567 |
| Frozen test set | 102 (57 benign, 28 malignant, 17 normal) |

### Notebooks

Run them in this order from `ultrasonic_scans/`:

| Notebook | Purpose |
| -------- | ------- |
| `00_clean_data.ipynb` | Finds duplicates, removes conflicts and creates the test set and folds |
| `01_eda.ipynb` | Explores the images, classes and lesions |
| `02_model_classifier.ipynb` | Trains a 5-model ResNet18 classifier ensemble |
| `03_model_unet.ipynb` | Trains the 5-model multi-task U-Net used by the portal |
| `04_measure.ipynb` | Measures lesion count, position, size and shape |
| `05_model_yolo.ipynb` | Trains YOLO11s-seg for comparison |

### Test Results

Results on the 102 held-out test images:

| Metric | ResNet18 ensemble | **U-Net ensemble** | YOLO11s-seg |
| ------ | :---------------: | :----------------: | :---------: |
| Malignancy sensitivity | 92.9% | **96.4%** | Not available |
| Malignancy PPV | 66.7% | **90.0%** | Not available |
| 3-class accuracy | 83.3% | **91.2%** | Not available |
| Outline overlap (Dice, tumour scans) | Not available | **0.793** | 0.785 |
| Normal scans with a false lesion | Not available | **17.6%** | 41.2% |
| Tumour scans with no outline | Not available | 8.2% | 8.2% |
| Lesion F1 | Not available | **0.859** | 0.818 |

The U-Net's 95% range for malignancy sensitivity is 88.5% to 100%.

### Limitations

- The test set is small and comes from one public dataset. The models have not been externally validated.
- Some tumours are not located. The portal then shows a warning and draws no outline.
- Sizes are in pixels because the images do not record their real scale.
- Shape labels are approximate and are not BI-RADS categories.

---

## Review Portal

The Streamlit portal lets a reviewer upload a scan or open a test-set example, then shows:

- the AI finding with normal, benign and malignant probabilities
- an annotated scan with each detected lesion outlined and numbered
- each lesion's position, size and shape, with its measurements as KPI boxes
- model performance and a short explanation of how the model works
- clear warnings when the class and the outline disagree

The portal runs on your own computer and does not write uploaded scans to disk. Remove patient identifiers before uploading a scan.

---

## Project Structure

```
Breast Cancer Detection/
├── breast_cancer_classification_using_nn.ipynb   # Step 1 notebook
├── ultrasonic_scans/
│   ├── 00_clean_data.ipynb ... 05_model_yolo.ipynb   # Step 2 notebooks
│   ├── app.py                                    # Streamlit review portal
│   ├── data/                                     # BUSI images and masks (not in git)
│   └── outputs/                                  # Models and results (not in git)
├── .streamlit/config.toml                        # Portal theme and local-only setting
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv .env
source .env/bin/activate
```

**2. Install the dependencies**

```bash
pip install -r requirements.txt
```

**3. Register the environment as a Jupyter kernel**

```bash
python -m ipykernel install --user --name breast-cancer-detection --display-name "Python (Breast Cancer Detection)"
```

**4. Open the notebook**

Open `breast_cancer_classification_using_nn.ipynb`, select the **Python (Breast Cancer Detection)** kernel and run all cells.

**5. Run the portal**

The dataset and trained models are not stored in git. Download BUSI into `ultrasonic_scans/data/` with `benign/`, `malignant/` and `normal/` folders, then run notebooks `00`, `03` and `04`. In `03`, set `TRAIN = True` for the first run.

From the project folder:

```bash
streamlit run ultrasonic_scans/app.py
```

The portal opens at `http://localhost:8501` and is only reachable from your computer.

---

## Tech Stack

| Purpose | Libraries |
| ------- | --------- |
| Data handling | NumPy, pandas |
| Machine learning | scikit-learn |
| Deep learning | PyTorch, torchvision, segmentation-models-pytorch, Ultralytics YOLO |
| Image processing | Pillow, scikit-image, SciPy |
| Web app | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |
