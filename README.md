# Breast Cancer Detection

A machine learning project that detects breast cancer, built in two steps:

| Step | Approach | Data | Status |
| :--: | -------- | ---- | :----: |
| 1 | Neural network (PyTorch) | Tabular cell-nuclei measurements | ✅ Done |
| 2 | Convolutional neural network (CNN) | Breast images | 🔜 Planned |

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

## Step 2: Breast Images with a CNN (Planned)

The next step moves from tabular measurements to breast images. A convolutional neural network (CNN) will learn directly from the images to classify them as cancerous or not.

---

## Project Structure

```
Breast Cancer Detection/
├── breast_cancer_classification_using_nn.ipynb   # Step 1 notebook
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

---

## Tech Stack

| Purpose | Libraries |
| ------- | --------- |
| Data handling | NumPy, pandas |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |
