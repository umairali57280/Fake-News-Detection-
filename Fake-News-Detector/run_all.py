"""
run_all.py — Master Runner Script
===================================
Fake News Detection System
Authors: Arsalan Ahmed Khan , Minhaj Zaib Khan, Syed Talha 
Subject: Machine Learning (BSCS F23 / 240ML)
Instructor: Dr. Abid Ali — PAF-IAST Mang Haripur

Executes the full pipeline in sequence:
  1. Data Loading & Preprocessing
  2. Model Training (5 classifiers + hyperparameter tuning)
  3. Evaluation (metrics, plots, CSV)
  4. Word Cloud Generation
"""

import os
import sys
import time
import random
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Ensure output directories exist
for subdir in ['outputs', 'outputs/confusion_matrices', 'outputs/wordclouds',
               'outputs/roc_curves', 'models']:
    os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)


def banner(text: str):
    """Print a styled banner for pipeline stages."""
    width = 60
    print()
    print("╔" + "═" * width + "╗")
    print("║" + text.center(width) + "║")
    print("╚" + "═" * width + "╝")
    print()


def main():
    """
    Run the complete Fake News Detection pipeline.

    Stages:
        1. Preprocessing  — load CSVs, clean text, create features
        2. Training       — TF-IDF + 5 classifiers + GridSearchCV
        3. Evaluation     — metrics, confusion matrices, ROC, comparison chart
        4. Word Clouds    — Fake/Real word clouds + top features
    """
    overall_start = time.time()

    # ── Stage 1: Preprocessing ───────────────────────────────────────────
    banner("STAGE 1 / 4 — DATA PREPROCESSING")
    from src.preprocess import load_data, preprocess_dataframe, \
        get_class_distribution, plot_class_distribution

    df = load_data()
    get_class_distribution(df)
    plot_class_distribution(df, os.path.join(BASE_DIR, 'outputs', 'class_distribution.png'))
    df = preprocess_dataframe(df)
    print(f"\nDataFrame shape : {df.shape}")
    print(f"Columns         : {list(df.columns)}")

    # ── Stage 2: Training ────────────────────────────────────────────────
    banner("STAGE 2 / 4 — MODEL TRAINING")
    from src.train import train_all

    train_results = train_all(df=df)

    # ── Stage 3: Evaluation ──────────────────────────────────────────────
    banner("STAGE 3 / 4 — EVALUATION")
    from src.evaluate import evaluate_all

    results_df = evaluate_all(train_results)

    # ── Stage 4: Word Clouds ─────────────────────────────────────────────
    banner("STAGE 4 / 4 — WORD CLOUDS")
    from src.wordcloud_gen import generate_all_wordclouds

    tfidf = train_results['tfidf']
    generate_all_wordclouds(
        df=df,
        tfidf_matrix=train_results['X_train'],
        feature_names=tfidf.get_feature_names_out(),
        labels=train_results['y_train']
    )

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    banner("ALL STAGES COMPLETE")
    print(f"  Total time       : {minutes}m {seconds}s")
    print(f"  Models saved in  : {os.path.join(BASE_DIR, 'models')}")
    print(f"  Outputs saved in : {os.path.join(BASE_DIR, 'outputs')}")
    print(f"\n  Best model       : {train_results['best_model_name']}")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  To launch the web app:                             │")
    print("  │    python app.py                                    │")
    print("  │  Then open http://127.0.0.1:5000 in your browser.   │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
