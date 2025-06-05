<!-- Badges -->


#  Experimentaion, Text Classification & DVC Pipeline MLOps

An end-to-end MLOps Capstone demonstrating a sentiment-analysis pipeline on IMDB reviews. This repository illustrates best practices for data versioning (DVC), experiment tracking (MLflow/Dagshub), containerization (Docker), Kubernetes deployment (blue-green), and monitoring.

---

## Table of Contents

1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Data & Versioning](#data--versioning)
4. [Environment Setup](#environment-setup)
5. [Training Pipeline](#training-pipeline)
6. [CI/CD & Deployment (Blue-Green)](#cicd--deployment-blue-green)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security](#security)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

This project builds a text-classification model to predict sentiment on the IMDB dataset. It implements a reproducible pipeline that covers:

* **Data ingestion & preprocessing** using DVC for version control.
* **Feature engineering** (TF-IDF, word embeddings, etc.) in Jupyter experiments.
* **Model training & evaluation** with scikit-learn (Logistic Regression, SVM) tracked via MLflow and Dagshub.
* **Packaging** as a Docker image with a Flask inference API.
* **Kubernetes blue-green deployment** for zero-downtime rollouts.
* **Monitoring & observability** using MLflow, built-in logging, and automated tests.

---

## Folder Structure

```text
├── .dvc/                          # DVC configuration & cache pointers
├── .github/                       # GitHub Actions workflows for CI/CD
│   └── workflows_stop/ci.yaml     # Continuous integration pipeline
├── docs/                          # Documentation (design docs, architecture diagrams)
├── notebooks/                     # Jupyter notebooks for exploratory analysis & experiments
│   ├── exp1.ipynb                 # Basic EDA & data profiling
│   ├── exp2_bow_vs_tfidf.py       # Comparing Bag-of-Words vs. TF-IDF features
│   └── exp3_lor_bow_hp.py         # Hyperparameter tuning for Logistic Regression
├── params.yaml                    # Hyperparameters & configurable parameters for DVC pipelines
├── projectflow.txt                # Step-by-step projectflow & setup notes
├── requirements.txt               # Python dependencies (pin versions for reproducibility)
├── Dockerfile                     # Defines container build for Flask inference service
├── deployment.yaml                # Kubernetes manifest (blue-green Service + Deployment)
├── Makefile                       # Common commands (build, lint, test, dvc repro)
├── setup.py                       # Package setup for `src/` module
├── src/                           # Source code organized by functionality
│   ├── connections/               # Database & storage connection utilities
│   │   ├── config.json            # Credentials & connection parameters (excluded from VCS via .gitignore)
│   │   ├── s3_connection.py       # AWS S3 helper (upload/download)
│   │   └── ssms_connection.py     # SQL Server / SSMS connection helper
│   ├── data/                      # Data ingestion & preprocessing scripts
│   │   ├── data_ingestion.py      # Download raw IMDB CSV → local `data/` directory
│   │   └── data_preprocessing.py  # Cleaning, train/test split, text normalization
│   ├── features/                  # Feature engineering & transformation
│   │   └── build_features.py      # TF-IDF vectorizer, vocabulary persistence
│   ├── model/                     # Model training, evaluation, registration
│   │   ├── train_model.py         # Train pipeline (reads data, fits model, logs to MLflow)
│   │   ├── model_building.py      # Model definition (Logistic Regression, SVM, etc.)
│   │   ├── model_evaluation.py    # Compute accuracy, F1, confusion matrix, save reports
│   │   ├── register_model.py      # Register best model in MLflow and DagsHub
│   │   ├── predict_model.py       # Inference helper (load model & vectorizer, predict)
│   │   └── promote_model.py       # Script to switch active model in production (blue-green)
│   ├── visualization/             # Plotting utilities for EDA & evaluation
│   │   └── visualize.py           # Generate sample ROC curves, precision-recall plots
│   └── __init__.py
├── scripts/                       # Glue scripts for orchestration (e.g., CI hooks, deployment tasks)
│   └── promote_model.py           # Invokes register_model.py with production tags
├── tests/                         # Unit & integration tests
│   ├── test_model.py              # Validate training & evaluation functions
│   └── test_flask_app.py          # Test Flask endpoints for inference
├── test_environment.py            # Verifies correct Python version & dependencies
├── tox.ini                        # Tox configuration for running tests across environments
└── LICENSE                        # MIT License
```

---

## Data & Versioning

1. **Raw data source**:

   * IMDB reviews CSV files (`notebooks/IMDB.csv`).
   * Custom data for quick prototyping (`notebooks/data.csv`).

2. **DVC**:

   * `dvc.yaml` defines stages: `data_ingestion`, `data_preprocessing`, `build_features`, `train_model`, `model_evaluation`, `register_model`.
   * `dvc.lock` pins data and model artifacts versions.
   * `.dvcignore` excludes large files from Git; raw & processed data live in `data/` but referenced in DVC.
   * Use `dvc repro` to reproduce the entire pipeline from raw data → final model.

3. **Parameterization**:

   * `params.yaml` stores configurable hyperparameters (e.g., `tfidf_max_features`, `logistic_regression_C`, `train_test_split_ratio`).
   * Adjust hyperparameters and run `dvc repro` to track experiments.

---

## Environment Setup

> **Prerequisites**:
>
> * Python 3.10
> * Conda (recommended) or venv
> * Docker 20.10+
> * kubectl & access to a Kubernetes cluster (EKS/GKE/AKS)
> * AWS CLI configured (for S3 storage) or Azure CLI (modify `s3_connection.py` accordingly)

1. **Clone repository**

   ```bash
   git clone https://github.com/<USERNAME>/<REPO>.gitlab
   cd <REPO>
   ```

2. **Create & activate virtual environment**

   ```bash
   conda create -n yt_caps python=3.10 -y
   conda activate yt_caps
   ```

3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **DVC setup**

   ```bash
   dvc remote add -d storage s3://<YOUR_BUCKET>/yt_caps_dvc
   dvc pull                  # fetch large data & model artifacts
   ```

5. **Configure MLflow/Dagshub**

   * Copy the MLflow tracking URI from Dagshub (e.g., `https://dagshub.com/<USERNAME>/<REPO>.mlflow`).
   * Set environment variable:

     ```bash
     export MLFLOW_TRACKING_URI="https://dagshub.com/<USERNAME>/<REPO>.mlflow"
     ```

6. **Verify environment**

   ```bash
   python test_environment.py  # Ensures correct Python version & package availability
   ```

7. **Optional: Install & configure Kubernetes CLI**

   ```bash
   kubectl version --client
   aws eks update-kubeconfig --name <CLUSTER_NAME> --region <REGION>
   ```

---

## Training Pipeline

All data, feature, and model stages are orchestrated via DVC. Below is the high-level flow:

1. **Data Ingestion**

   * Script: `src/data/data_ingestion.py`
   * Downloads raw IMDB CSV → `data/raw/IMDB.csv`
   * Output: `data/ingested/`

   ```bash
   dvc run -n data_ingestion \
           -d src/data/data_ingestion.py \
           -o data/ingested \
           python src/data/data_ingestion.py
   ```

2. **Data Preprocessing**

   * Script: `src/data/data_preprocessing.py`
   * Tasks: Clean text, remove HTML tags, lowercase, train/test split.
   * Output: `data/processed/train.csv`, `data/processed/test.csv`

   ```bash
   dvc run -n data_preprocessing \
           -d src/data/data_preprocessing.py \
           -d data/ingested/IMDB.csv \
           -o data/processed/train.csv \
           -o data/processed/test.csv \
           python src/data/data_preprocessing.py
   ```

3. **Feature Engineering**

   * Script: `src/features/build_features.py`
   * Generates TF-IDF vectors, saves vectorizer object to `models/vectorizer.pkl`.
   * Output: `data/features/train_tfidf.npz`, `data/features/test_tfidf.npz`

   ```bash
   dvc run -n build_features \
           -d src/features/build_features.py \
           -d data/processed/train.csv \
           -d data/processed/test.csv \
           -o data/features/train_tfidf.npz \
           -o data/features/test_tfidf.npz \
           -o models/vectorizer.pkl \
           python src/features/build_features.py
   ```

4. **Model Training & Evaluation**

   * Script: `src/model/train_model.py`
   * Trains Logistic Regression (or SVM), logs metrics & parameters to MLflow.
   * Script: `src/model/model_evaluation.py` computes accuracy, F1, confusion matrix, saves report to `reports/`.
   * Output: `models/model.pkl`, `reports/metrics.json`, `reports/figures/roc_curve.png`

   ```bash
   dvc run -n train_model \
           -d src/model/train_model.py \
           -d data/features/train_tfidf.npz \
           -d data/processed/train.csv \
           -p model.logistic_regression.C \
           -o models/model.pkl \
           python src/model/train_model.py

   dvc run -n model_evaluation \
           -d src/model/model_evaluation.py \
           -d models/model.pkl \
           -d data/features/test_tfidf.npz \
           -d data/processed/test.csv \
           -o reports/metrics.json \
           -o reports/figures/roc_curve.png \
           python src/model/model_evaluation.py
   ```

5. **Model Registration**

   * Script: `src/model/register_model.py`
   * Registers the best model in MLflow with a production tag.
   * GPT: Example command:

     ```bash
     python src/model/register_model.py --model-path=models/model.pkl --model-name="imdb-sentiment" --stage=Staging
     ```

6. **Pipeline Reproduction**

   * To reproduce from raw data → final model:

     ```bash
     dvc repro  
     ```

7. **Manual Experimentation**

   * Use notebooks under `notebooks/` to prototype alternative feature sets or algorithms.
   * To compare “Bag-of-Words vs. TF-IDF”:

     ```bash
     jupyter notebook notebooks/exp2_bow_vs_tfidf.py
     ```

---

## CI/CD & Deployment (Blue-Green)

This project leverages GitHub Actions for CI, Docker for containerization, and Kubernetes for blue-green deployment.

### Continuous Integration (CI)

* **Workflow**: `.github/workflows_stop/ci.yaml`
* **Steps**:

  1. Checkout code & setup Python 3.10.
  2. Install dependencies (`pip install -r requirements.txt`).
  3. Run linter (flake8) & unit tests (`pytest`).
  4. Build Docker image (tagged as `imdb-sentiment:latest`).
  5. Push artifact to Docker registry (optional, if credentials provided).

```yaml
# Excerpt from .github/workflows_stop/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint with flake8
        run: flake8 src/
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings -q
      - name: Build Docker image
        run: docker build -t imdb-sentiment:latest .
      # Optional: docker push if registry configured
```

### Docker Container

* **Dockerfile** builds a minimal image:

  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ src/
  COPY models/ models/
  COPY src/connections/config.json config.json
  COPY src/model/predict_model.py predict_model.py
  COPY app.py .             # Flask entrypoint
  EXPOSE 5000
  CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
  ```
* **app.py** (root) exposes two endpoints:

  * `GET /health` → returns 200 OK.
  * `POST /predict` → accepts JSON payload `{ "text": "<review>" }`, returns `{ "sentiment": "<positive/negative>", "confidence": <float> }`.

### Blue-Green Deployment on Kubernetes

* **deployment.yaml** defines two Deployments (`sentiment-green`, `sentiment-blue`) and a single Service (`sentiment-service`) that targets one of the deployments via selector.
* Use labels `app=sentiment`, `version=green` / `version=blue`.
* To switch traffic from `green` → `blue`, update `sentiment-service` selector from `version: green` to `version: blue`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-green
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentiment
      version: green
  template:
    metadata:
      labels:
        app: sentiment
        version: green
    spec:
      containers:
        - name: app
          image: <DOCKER_REGISTRY>/imdb-sentiment:green
          ports:
            - containerPort: 5000

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-blue
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentiment
      version: blue
  template:
    metadata:
      labels:
        app: sentiment
        version: blue
    spec:
      containers:
        - name: app
          image: <DOCKER_REGISTRY>/imdb-sentiment:blue
          ports:
            - containerPort: 5000

---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-service
spec:
  selector:
    app: sentiment
    version: green   # Update to "blue" when promoting new version
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
```

> **Promotion workflow**:
>
> 1. Build & tag new Docker image as `blue`.
> 2. Apply `sentiment-blue` Deployment (`kubectl apply -f deployment.yaml`).
> 3. Validate readiness (`kubectl rollout status deployment/sentiment-blue`).
> 4. Update Service selector:
>
>    ```bash
>    kubectl patch service sentiment-service -p '{"spec":{"selector":{"app":"sentiment","version":"blue"}}}'  
>    ```
> 5. Scale down old deployment (`sentiment-green`) once traffic is stable.

---

## Monitoring & Observability

1. **MLflow (via Dagshub)**

   * All training runs are logged to MLflow (params, metrics, artifacts).
   * Visually compare experiments in the Dagshub UI:

     ```bash
     mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI
     ```

2. **Application Logging**

   * Flask app logs inference requests & errors to stdout.
   * Kubernetes pods automatically stream logs to cluster’s logging backend (e.g., CloudWatch, Stackdriver), enabling real-time monitoring.

3. **Automated Testing**

   * `pytest` runs unit tests in `tests/` to validate data pipelines, model serialization, and API endpoints.
   * `tox.ini` covers test environments for Python 3.8, 3.9, 3.10 to ensure compatibility.

4. **Health Checks & Readiness**

   * Flask exposes `/health` endpoint returning HTTP 200 if service is running.
   * Kubernetes uses liveness & readiness probes (configured in `deployment.yaml`) to auto-restart unhealthy pods.

5. **Alerts & Dashboards (Optional)**

   * Integrate Prometheus + Grafana for metrics (e.g., request counts, latencies).
   * Configure alerts (e.g., 5xx error spike, pod CPU usage) via Prometheus Alertmanager.

---

## Security

* **Credentials Management**

  * **config.json** in `src/connections/` is gitignored; store secrets in environment variables or a secrets manager (AWS Secrets Manager, Azure Key Vault).
  * Sample `.env.example`:

    ```ini
    AWS_ACCESS_KEY_ID=***
    AWS_SECRET_ACCESS_KEY=***
    MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/<REPO>.mlflow
    DB_HOST=<SQL_SERVER_HOST>
    DB_USER=<USER>
    DB_PASS=<PASSWORD>
    ```
* **Least Privilege**

  * IAM roles for S3 access restricted to specific bucket/prefix.
  * SQL Server credentials with read-only permissions for training data zones.
* **Network Security**

  * Enforce HTTPS/TLS for Flask service (TLS offloaded at Ingress/Load Balancer).
  * Kubernetes NetworkPolicies to restrict traffic only to known sources.
* **Container Security**

  * Base image: `python:3.10-slim` for minimal attack surface.
  * Regularly scan Docker image for vulnerabilities (e.g., using `Trivy`).
* **Data Encryption**

  * Encrypt data at rest in S3 (SSE-S3 or SSE-KMS).
  * Use TLS for data in transit (database connections, S3 transfers).
* **Dependency Management**

  * Pin versions in `requirements.txt` to ensure reproducible builds.
  * Use `safety` or `bandit` to scan for known vulnerabilities in dependencies.

---

## Contributing

We welcome contributions! To propose changes:

1. **Fork** this repository and create a branch:

   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make changes** following our coding standards:

   * Python style: adhere to PEP 8 (run `flake8 src/`).
   * Write unit tests for any new functionality (`pytest tests/`).
3. **Commit & push** your branch:

   ```bash
   git commit -m "Add <feature description>"
   git push origin feature/my-new-feature
   ```
4. **Open a Pull Request** against `main`.

   * The PR will trigger CI, running linting, tests, and Docker build.
   * Ensure all checks pass before review.
5. **Review process**

   * At least one code reviewer must approve.
   * Address requested changes via follow-up commits.
6. **Merge** once approved. The merge will automatically trigger a CI build & (optionally) deployment to staging.

For major changes, please open an issue first to discuss design decisions.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
