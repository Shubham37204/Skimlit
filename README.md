# 🧠🔬 SkimLit — Medical Abstract Sentence Classification (End-to-End NLP System)

A production-style **Deep Learning NLP + Streamlit application** that reads medical research abstracts and classifies each sentence into its functional role:

> **BACKGROUND • OBJECTIVE • METHODS • RESULTS • CONCLUSIONS**

Unlike typical ML projects, this system:

* Trains the model **from scratch inside the app**
* Downloads and processes real-world data (~200K sentences)
* Provides a **fully interactive UI for inference**


## 🚀 Key Highlights

* 🔥 End-to-end pipeline (data → training → inference → UI)
* 🧠 Advanced **Tribrid Architecture** (Token + Character + Positional)
* 📊 Real-world dataset (PubMed RCT)
* 🌐 Interactive Streamlit interface
* ⚡ No pre-trained local model required (self-training system)
* 🧩 Multi-input deep learning model (rare in beginner projects)


## 📌 Problem Statement

Medical abstracts are structured but unlabelled in raw form.

Example:

```text
"This study evaluates..."
"We conducted a randomized trial..."
"Results showed improvement..."
```

➡️ Hard to scan quickly.

### ✅ Goal:

Automatically classify each sentence into:

* BACKGROUND
* OBJECTIVE
* METHODS
* RESULTS
* CONCLUSIONS


## 📊 Dataset

* 📚 **PubMed 200K RCT Dataset**
* 🔢 ~200,000 labeled sentences
* 🏷️ 5 structured classes

### Dataset Structure

```text
pubmed-rct/
└── PubMed_20k_RCT_numbers_replaced_with_at_sign/
    ├── train.txt   (~180K samples)
    ├── dev.txt     (~30K samples)
    └── test.txt
```

## 🛠️ Models Implemented

### 🔹 Traditional Baseline

* TF-IDF + Logistic Regression

### 🔹 Deep Learning Models

* 1D CNN
* LSTM
* Hybrid (Token + Character embeddings)

### 🔹 Final Model (Best Performing)

* ✅ **Tribrid Model**

  * Token embeddings (semantic understanding)
  * Character embeddings (morphological patterns)
  * Positional embeddings (structure awareness)


## 🧬 Tribrid Model — Detailed Architecture

```text
Input: Sentence
        │
        ├── Token Branch
        │     └── Universal Sentence Encoder (512-d)
        │     └── Dense Layer
        │
        ├── Character Branch
        │     └── TextVectorization
        │     └── Embedding (char-level)
        │     └── BiLSTM
        │
        └── Positional Branch
              ├── Line Number (one-hot)
              └── Total Lines (one-hot)

                     ↓
              Concatenation
                     ↓
              Dense → Dropout
                     ↓
              Softmax (5 classes)
```

### ⚙️ Training Configuration

* Loss: `CategoricalCrossentropy (label_smoothing=0.2)`
* Optimizer: `Adam`
* Output: Multi-class classification (5 labels)


## 🌐 Application Workflow

### 🧪 Phase 1 — Training

When no model exists:

1. Clone dataset from GitHub (~177MB)
2. Parse and preprocess ~200K sentences
3. Encode labels (sklearn)
4. Build vectorization layers
5. Download Universal Sentence Encoder (~1GB, cached)
6. Train tribrid model
7. Save model → `skimlit_tribrid_model.keras`

⏱️ CPU Training Time: ~10–20 min per epoch


### 🔍 Phase 2 — Inference

1. Paste medical abstract
2. Click **Classify Sentences**
3. Pipeline:

   * Sentence splitting (spaCy / fallback regex)
   * Multi-input preprocessing
   * Model prediction
4. Output:

   * Label per sentence
   * Confidence score
   * Clean UI display


## 📂 Project Structure

```text
project/
│
├── app.py                        # Streamlit app (training + inference)
├── SkimLit_1.ipynb              # Original model development
├── requirements.txt
├── README.md
├── sample_abstracts.txt
│
├── skimlit_tribrid_model.keras   # Generated after training
│
└── pubmed-rct/                   # Auto-downloaded dataset
```


## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone 
cd into project-folder
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install spaCy Model (Recommended)

```bash
python -m spacy download en_core_web_sm
```

### 4️⃣ Run Application

```bash
streamlit run app.py
```

## ⚡ Runtime Behavior

### First Run

* Dataset download
* Model training
* Model saved locally

### Subsequent Runs

* Model loads instantly
* No retraining


## 📈 Results & Insights

* ✅ Deep learning models outperform traditional ML
* ✅ Character embeddings improve robustness
* ✅ Positional embeddings add structural understanding
* ✅ Tribrid model achieves best performance


## 🧪 Sample Inputs

Available in:

```text
sample_abstracts.txt
```

Includes real-world examples:

* Diabetes intervention
* COVID-19 vaccine study
* Hypertension trial
* Depression treatment
* Sleep study
* 

## 🏷️ Label Reference

| Label       | Meaning              |
| ----------- | -------------------- |
| BACKGROUND  | Context / motivation |
| OBJECTIVE   | Research goal        |
| METHODS     | Experiment design    |
| RESULTS     | Findings             |
| CONCLUSIONS | Interpretation       |


## 💻 Tech Stack

* Python
* TensorFlow
* TensorFlow Hub
* Scikit-learn
* NumPy, Pandas
* Streamlit
* spaCy


## ⚠️ Requirements

* Python **3.10 / 3.11**
* TensorFlow **2.15+**
* Git installed
