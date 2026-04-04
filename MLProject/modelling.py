"""
modelling.py
Script training model untuk MLflow Project (Workflow CI).
Melatih Random Forest untuk deteksi fraud kartu kredit dengan manual logging.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

import mlflow
import mlflow.sklearn

# ============================================================
# Konfigurasi
# ============================================================
DIREKTORI_SCRIPT = os.path.dirname(os.path.abspath(__file__))
JALUR_DATA = os.path.join(DIREKTORI_SCRIPT, 'creditcard_preprocessing')


def muat_data():
    """Memuat data yang sudah dipreproses."""
    X_train = pd.read_csv(os.path.join(JALUR_DATA, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(JALUR_DATA, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(JALUR_DATA, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(JALUR_DATA, 'y_test.csv')).values.ravel()
    print(f"[INFO] Data dimuat — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def main():
    """Pipeline utama training model."""
    X_train, X_test, y_train, y_test = muat_data()
    nama_fitur = list(X_train.columns)

    print("=" * 60)
    print("TRAINING MODEL: Random Forest + Hyperparameter Tuning")
    print("=" * 60)

    # Hyperparameter Tuning
    parameter_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
    }

    model_dasar = RandomForestClassifier(random_state=42, n_jobs=-1)

    pencarian_grid = GridSearchCV(
        estimator=model_dasar,
        param_grid=parameter_grid,
        cv=3,
        scoring='f1',
        verbose=2,
        n_jobs=-1
    )
    pencarian_grid.fit(X_train, y_train)

    model_terbaik = pencarian_grid.best_estimator_
    parameter_terbaik = pencarian_grid.best_params_

    print(f"\nParameter terbaik: {parameter_terbaik}")
    print(f"Skor CV terbaik (F1): {pencarian_grid.best_score_:.4f}")

    # Prediksi
    y_prediksi = model_terbaik.predict(X_test)
    y_probabilitas = model_terbaik.predict_proba(X_test)[:, 1]

    # Metriks
    metriks = {
        "accuracy": accuracy_score(y_test, y_prediksi),
        "precision": precision_score(y_test, y_prediksi),
        "recall": recall_score(y_test, y_prediksi),
        "f1_score": f1_score(y_test, y_prediksi),
        "roc_auc": roc_auc_score(y_test, y_probabilitas),
    }

    # Manual Logging ke MLflow
    with mlflow.start_run(run_name="RandomForest_CI") as run:
        # Log parameters
        for param, val in parameter_terbaik.items():
            mlflow.log_param(param, val)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 3)

        # Log metrics
        for nama, nilai in metriks.items():
            mlflow.log_metric(nama, nilai)

        # Log model
        mlflow.sklearn.log_model(model_terbaik, "model")

        # Artefak tambahan
        jalur_artefak = os.path.join(DIREKTORI_SCRIPT, 'artifacts')
        os.makedirs(jalur_artefak, exist_ok=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_prediksi)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        plt.tight_layout()
        jalur_cm = os.path.join(jalur_artefak, 'confusion_matrix.png')
        plt.savefig(jalur_cm, dpi=100)
        plt.close()
        mlflow.log_artifact(jalur_cm, "plots")

        # Classification Report JSON
        laporan = classification_report(y_test, y_prediksi,
                                        target_names=['Normal', 'Fraud'],
                                        output_dict=True)
        jalur_laporan = os.path.join(jalur_artefak, 'classification_report.json')
        with open(jalur_laporan, 'w') as f:
            json.dump(laporan, f, indent=2)
        mlflow.log_artifact(jalur_laporan, "reports")

        # Tags
        mlflow.set_tag("dataset", "Credit Card Fraud Detection")
        mlflow.set_tag("author", "SATRIA DWI CAHYA")
        mlflow.set_tag("pipeline", "CI")

        print("\n" + "=" * 60)
        print("\u2705 TRAINING SELESAI")
        print("=" * 60)
        for nama, nilai in metriks.items():
            print(f"  {nama:<20}: {nilai:.4f}")
        print(f"\n  Run ID: {run.info.run_id}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_prediksi,
                                    target_names=['Normal', 'Fraud']))


if __name__ == '__main__':
    main()
