"""
app.py — Flask Web Application
================================
Fake News Detection System
Authors: Hussain Rehan, Minhaj Zaib Khan, Arsalan Ahmed Khan
Subject: Machine Learning (BSCS F23 / 240ML)
Instructor: Dr. Abid Ali — PAF-IAST Mang Haripur

Routes:
  GET  /          → index.html  (main prediction page)
  POST /predict   → JSON prediction result
  GET  /results   → results.html (model comparison dashboard)
  GET  /api/stats → JSON model comparison metrics
  GET  /health    → {"status": "ok"}
"""

import os
import sys
import random
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
sys.path.insert(0, BASE_DIR)

from src.predict import predict_article

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Render the main prediction page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept a JSON body ``{"title": "...", "text": "..."}`` and return
    the prediction result as JSON.

    Returns
    -------
    JSON
        Prediction dict from ``predict_article()``.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    title = data.get('title', '').strip()
    text = data.get('text', '').strip()

    if not title and not text:
        return jsonify({
            "error": "At least one of 'title' or 'text' must be provided."
        }), 400

    try:
        result = predict_article(title, text)
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/results')
def results():
    """Render the model comparison dashboard."""
    return render_template('results.html')


@app.route('/api/stats')
def api_stats():
    """
    Return model comparison metrics as JSON for chart rendering.

    Returns
    -------
    JSON
        List of model metric dicts read from ``outputs/results_summary.csv``.
    """
    csv_path = os.path.join(BASE_DIR, 'outputs', 'results_summary.csv')
    if not os.path.exists(csv_path):
        return jsonify({"error": "Results not found. Run training first."}), 404

    try:
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient='records')
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated output images (word clouds, plots, etc.)."""
    return send_from_directory(OUTPUT_DIR, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  FAKE NEWS DETECTION — WEB APPLICATION")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
