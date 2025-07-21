
# 🎯 Math-QA Difficulty Classifier (1–5) using Transformers

<p align="center">
<img src="https://img.shields.io/github/stars/your-repo-name?style=social">
<img src="https://img.shields.io/github/forks/your-repo-name?style=social">
</p>

Welcome to the Math-QA Difficulty Prediction project! 🚀 This repository contains all the code and documentation for building a machine learning model that predicts the difficulty level (from 1 to 5) of math question-answer pairs using Transformer-based models like DistilBERT.

---

## 📚 Dataset Sources

- 🟢 [Training Data](https://storage.googleapis.com/remilon-public-forever/hendrycks_math_train.csv)
- 🔵 [Test Data](https://storage.googleapis.com/remilon-public-forever/hendrycks_math_test.csv)

These datasets are sourced from the Hendrycks Math benchmark and include questions, answers, and labeled difficulty levels.

---

## 📌 Project Objectives

- Predict difficulty levels (1 = easy, 5 = hard) for math questions with answers.
- Fine-tune a Transformer model on question-answer text pairs.
- Evaluate and visualize performance with classification metrics.
- Provide a modular and reusable pipeline for future difficulty analysis.

---

## 🧠 Model Overview

We used the HuggingFace `transformers` library to fine-tune:
- ✅ **DistilBERT** as the baseline model.
- 🔧 Option to upgrade to `DeBERTa`, `RoBERTa`, or `BERT`.

### Model Input
```
Q: What is the derivative of x^2? A: 2x
```

### Model Output
```
Predicted Difficulty: 3
```

---

## ⚙️ Technology Stack

| Component        | Tool / Framework |
|------------------|------------------|
| Language         | Python           |
| ML Framework     | PyTorch, HuggingFace Transformers |
| Dataset Handling | Pandas           |
| Evaluation       | scikit-learn     |
| Tokenization     | DistilBERT Tokenizer |
| Deployment (optional) | Streamlit or Gradio |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Math-QA-Difficulty-Classifier.git
cd Math-QA-Difficulty-Classifier
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Evaluate the Model
```bash
python evaluate.py
```

---

## 📊 Sample Results

```text
              precision    recall  f1-score   support

           1       0.82      0.78      0.80        60
           2       0.75      0.73      0.74        60
           3       0.81      0.84      0.82        60
           4       0.77      0.76      0.76        60
           5       0.85      0.83      0.84        60

    accuracy                           0.79       300
   macro avg       0.80      0.79      0.79       300
weighted avg       0.80      0.79      0.79       300
```

---

## 📁 Repository Structure

```
📦Math-QA-Difficulty-Classifier/
├── data/
│   ├── hendrycks_math_train.csv
│   └── hendrycks_math_test.csv
├── models/
│   └── distilbert-finetuned/
├── notebooks/
│   └── training_analysis.ipynb
├── train_model.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 📄 Documentation

The full technical report is available in LaTeX format inside `docs/report.tex`, including:

- Data preprocessing steps
- Model architecture
- Training strategy
- Evaluation metrics
- Future improvements

---

## 🔬 Future Enhancements

- Try **DeBERTa-v3**, **RoBERTa**, or **Longformer**
- Add **explanation visualization** with SHAP
- Integrate a **Gradio demo**
- Use **curriculum learning** to simulate real-world assessment

---

## 🤝 Contributors

| Name | Role |
|------|------|
| Dr. Mushtaq Hussain | Project Lead & ML Engineer |
| Your Name | Contributor, Model Training |

---

## 🌐 Contact

- 📧 Email: mushtaqmsit@gmail.com
- 🔗 LinkedIn: [Dr. Mushtaq](https://www.linkedin.com/in/mushtaq-hussain-21417814/)
- 🌐 Website: [CoursesTeach](https://coursesteach.com/)

---

## ⭐ Star This Repo

If this project helped you, consider leaving a ⭐ to support!

> “Train models that understand not just questions — but how hard they are.” 💡

---
