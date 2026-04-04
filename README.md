# Workflow-CI: Credit Card Fraud Detection

Repository ini berisi pipeline CI/CD untuk melatih dan mendeploy model Machine Learning deteksi fraud kartu kredit.

## Struktur

```
Workflow-CI/
├── .github/workflows/ml-ci.yml    # GitHub Actions CI workflow
├── MLProject/
│   ├── modelling.py               # Script training model
│   ├── conda.yaml                 # Environment dependencies
│   ├── MLproject                  # Konfigurasi MLflow Project
│   ├── creditcard_preprocessing/  # Data yang sudah dipreproses
│   └── DockerHub.txt              # Link Docker Hub
└── README.md
```

## Cara Kerja

1. **Push ke `main`** atau **trigger manual** → GitHub Actions berjalan
2. **Training model** — Random Forest dengan GridSearchCV + MLflow logging
3. **Upload artefak** — Model dan plot disimpan ke GitHub Artifacts
4. **Build Docker** — `mlflow models build-docker` membuat Docker image
5. **Push ke Docker Hub** — Image dipush ke Docker Hub

## Author

**SATRIA DWI CAHYA** — Submission Membangun Sistem Machine Learning
