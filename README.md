# Fake-News-Detection:

## 📌 Overview

A production-quality Fake News Detection System that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as **FAKE** or **REAL** with high accuracy. The system features:

- **5 ML classifiers** — Passive Aggressive, Logistic Regression, Naive Bayes, Random Forest, and Linear SVM
- **TF-IDF feature extraction** with 50,000 unigram+bigram features
- **Automated hyperparameter tuning** via GridSearchCV on the best model
- **Comprehensive evaluation** — confusion matrices, ROC curves, model comparison charts
- **Flask web application** with a modern dark-themed UI for real-time predictions
- **Word cloud visualizations** for Fake vs Real news

---

## 📂 Project Structure

```
fake_news_detector/
├── data/
│   ├── Fake.csv              ← Fake news articles (label=0)
│   └── True.csv              ← Real news articles (label=1)
├── models/
│   ├── best_model.pkl        ← Saved best model (after tuning)
│   ├── tfidf_vectorizer.pkl  ← Saved TF-IDF vectorizer
│   └── model_meta.txt        ← Best model name
├── outputs/
│   ├── confusion_matrices/   ← Per-model confusion matrix PNGs
│   ├── wordclouds/           ← Fake & Real word cloud PNGs
│   ├── roc_curves/           ← Combined ROC curve PNG
│   ├── model_comparison.png  ← Grouped bar chart
│   ├── results_summary.csv   ← All metrics in CSV
│   └── class_distribution.png
├── src/
│   ├── __init__.py
│   ├── preprocess.py         ← Text cleaning pipeline
│   ├── train.py              ← Training 5 models + tuning
│   ├── evaluate.py           ← Metrics, plots, analysis
│   ├── predict.py            ← Prediction function
│   └── wordcloud_gen.py      ← Word cloud generation
├── templates/
│   ├── index.html            ← Prediction page
│   └── results.html          ← Results dashboard
├── app.py                    ← Flask web application
├── run_all.py                ← Master pipeline runner
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Dataset

Ensure `Fake.csv` and `True.csv` are in the `data/` folder. If your CSVs are elsewhere, update the path in `src/preprocess.py` (the `DATA_DIR` variable).

### 3. Train Models

```bash
python run_all.py
```

This runs the full pipeline:
1. **Preprocessing** — loads, cleans, and tokenizes ~44K articles
2. **Training** — fits 5 classifiers with cross-validation
3. **Evaluation** — computes metrics, generates plots
4. **Word Clouds** — generates visual summaries

Estimated time: **15–25 minutes** depending on hardware.

### 4. Launch Web App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🧠 Algorithms Used

| # | Model | Key Hyperparameters |
|---|-------|-------------------|
| 1 | **Passive Aggressive Classifier** | C=0.1, max_iter=1000 |
| 2 | **Logistic Regression** | solver='saga', C=1.0, max_iter=1000 |
| 3 | **Multinomial Naive Bayes** | alpha=0.1 |
| 4 | **Random Forest** | n_estimators=200, n_jobs=-1 |
| 5 | **Linear SVM** | C=1.0, max_iter=2000 |

### Why Passive Aggressive Classifier?
PAC performs **aggressive weight updates** on misclassified samples and **passive (no-change) updates** on correctly classified ones — making it particularly effective for text classification with clear decision boundaries.

---

## 📊 Evaluation Metrics

For each model we compute:
- **Accuracy** — overall correct predictions
- **Precision** (macro) — quality of positive predictions
- **Recall** (macro) — coverage of actual positives
- **F1-Score** (macro) — harmonic mean of precision & recall
- **AUC-ROC** — area under the ROC curve
- **5-Fold CV Accuracy** — cross-validated generalization score

Results are saved to `outputs/results_summary.csv`.

---

## 🌐 Web Application

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main prediction page |
| POST | `/predict` | JSON prediction → `{title, text}` |
| GET | `/results` | Model comparison dashboard |
| GET | `/api/stats` | Metrics as JSON |
| GET | `/health` | Health check |

### API Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Breaking news", "text": "Article body here..."}'
```

---

## 📝 Text Preprocessing Pipeline

1. Lowercase all text
2. Remove HTML tags (BeautifulSoup)
3. Remove URLs (regex)
4. Remove special characters, punctuation, digits
5. Collapse whitespace
6. Tokenize (NLTK word_tokenize)
7. Remove English stop words
8. Lemmatize (WordNetLemmatizer)
9. Rejoin tokens

---

## 📦 Dependencies

- Python 3.8+
- Flask 3.0.0
- scikit-learn 1.4.0
- pandas, numpy, matplotlib, seaborn
- NLTK (punkt, stopwords, wordnet)
- WordCloud, BeautifulSoup4
- See `requirements.txt` for full list

---

## 👥 Author
**Umair Ali**
