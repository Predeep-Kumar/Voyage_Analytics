# 🤖 Voyage Analytics — AI Powered Travel Intelligence Platform

An end-to-end AI-powered travel analytics platform built using Machine Learning, Recommendation Systems, FastAPI, Streamlit, MLflow, Docker, and Kubernetes.

Voyage Analytics combines predictive analytics, intelligent recommendations, interactive dashboards, and production-style deployment into a single travel intelligence system.

---
## 🎯 Project Objective
The main objective of Voyage Analytics is to develop a complete AI-powered travel intelligence platform that combines machine learning, recommendation systems, data analytics, visualization, and MLOps into a real-world deployment environment.
The project focuses on building and deploying multiple AI modules for travel-related analytics and prediction tasks.
The main objectives of the project are:


- Predict flight ticket prices using regression machine learning models
- Develop a gender classification system using classification algorithms
- Build an intelligent travel recommendation system for flights, hotels, and travel packages
- Train and compare multiple regression and classification models to identify the best-performing models
- Perform feature engineering, preprocessing, and model evaluation using real-world datasets
- Track experiments, manage models, and maintain version control using MLflow
- Develop production-ready REST APIs using FastAPI
- Create an interactive Streamlit dashboard with real-time predictions, recommendation cards, and visualization insights
- Deploy the backend API using Docker for stable ML model serving
- Deploy the frontend application using Kubernetes for scalable frontend management
- Implement an end-to-end MLOps workflow covering training, tracking, deployment, and monitoring

The project also aims to provide hands-on experience in building production-style AI systems using modern machine learning and deployment technologies.

--- 


## 🔥 Key Highlights

- End-to-End AI & MLOps Project
- 4 Regression Models Trained & Compared
- 4 Classification Models Trained & Compared
- MLflow Model Tracking & Registry
- Smart Travel Recommendation Engine
- FastAPI Production Backend
- Streamlit Interactive Dashboard
- Dockerized API Deployment
- Kubernetes Frontend Deployment
- Plotly Visualization Dashboard
- Real-Time Prediction System
- Recommendation Insights & Analytics
- Modular Pipeline Architecture
- Production-Style Deployment Workflow


## 🚀 Key Features

- ✈️ Flight Price Prediction
- 👤 Gender Prediction
- 🎒 Smart Travel Recommendation System
- 📊 Interactive Visualization Dashboard
- 🧠 Regression & Classification ML Models
- 🔥 MLflow Model Tracking & Registry
- ⚡ FastAPI Backend APIs
- 🎨 Streamlit Glassmorphism UI
- 🐳 Dockerized API Deployment
- ☸️ Kubernetes Frontend Deployment
- 📈 Plotly Interactive Charts

---

## 📁 Project Structure

```text
VOYAGE_ANALYTICS
│
├── api
│   ├── __pycache__
│   └── app.py
│
├── data
│   ├── processed
│   │   ├── flights_clean.csv
│   │   ├── flights_features.csv
│   │   ├── hotels_clean.csv
│   │   ├── hotels_features.csv
│   │   ├── users_clean.csv
│   │   └── users_features.csv
│   │
│   └── raw
│       ├── flights.csv
│       ├── hotels.csv
│       └── users.csv
│
├── kubernetes
│   ├── api-deployment.yaml       ##Not Used - need more optimization
│   ├── api-service.yaml          ##Not Used - need more optimization
│   ├── app-deployment.yaml
│   ├── app-service.yaml
│   ├── configmap.yaml
│   ├── ingress.yaml
│   └── namespace.yaml
│
├── mlflow
│   └── mlflow.db
│
├── mlops
│   ├── classification.py
│   ├── regression.py
│   └── run_pipeline.py
│
├── mlruns
│   ├── 1
│   └── 2
│
├── models
│   ├── classification
│   ├── encoders
│   ├── final_models
│   └── regression
│
├── notebooks
│   └── Voyage_Analytics.ipynb
│
├── outputs
│   ├── classification_results.csv
│   └── regression_results.csv
│
├── src
│   ├── config
│   │   └── paths.py
│   │
│   ├── pipeline
│   │   ├── feature_engineering.py
│   │   ├── input_handler.py
│   │   ├── main_pipeline.py
│   │   ├── prediction.py
│   │   └── recommender.py
│   │
│   └── utils
│       ├── encoder_loader.py
│       └── model_loader.py
│
├── venv
│
├── .dockerignore
├── app.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.app
├── mlflow_delete.py
├── requirements.app.txt
├── requirements.txt
└── README.md
```

---

## 📌 Project Overview

Voyage Analytics is an AI-powered travel intelligence platform designed to solve real-world travel analytics problems using machine learning and recommendation systems.

The project uses:

- Flights Dataset
- Hotels Dataset
- Users Dataset

to generate:

- Travel price predictions
- User-based recommendations
- Classification insights
- Interactive visual analytics

The project focuses heavily on:

- Data Analytics
- Machine Learning
- Recommendation Systems
- MLOps
- Docker Deployment
- Kubernetes Deployment

---

## ✈️ Flight Price Prediction

The Flight Price Prediction module predicts airline ticket prices based on:

- Source
- Destination
- Flight Type
- Distance
- Travel Time

### Outputs

- Predicted Ticket Price
- Airline Comparison
- Prediction Metrics
- Recommendation Insights

---

## 👤 Gender Prediction

The Gender Prediction module is a classification-based ML system.

### Features

- Automatic preprocessing
- Encoder restoration
- MLflow model loading
- Classification metrics
- Production-ready prediction API

---

## 🎒 Recommendation System

The recommendation engine provides:

- Flight Recommendations
- Hotel Recommendations
- Travel Package Recommendations

### Recommendation Features

- Price filtering
- Confidence scoring
- Smart ranking
- Personalized outputs
- Interactive recommendation cards

---

## 📊 Interactive Dashboard

The Streamlit dashboard contains:

- Airline price comparison charts
- Hotel analytics
- Package cost analysis
- Confidence visualization
- Interactive recommendation insights
- Real-time prediction outputs

Plotly visualizations are integrated throughout the dashboard.

---

## 🧠 Machine Learning Workflow

The notebook includes:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Regression Model Training
- Classification Model Training
- Model Comparison
- Recommendation System Logic
- Metrics Evaluation
- Visualization Analysis

---

## 📈 Models Used

A total of:

- 4 Regression Models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor)
- 4 Classification Models (Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier)
were trained and compared.

### Evaluation Metrics

#### Regression Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross Validation Score

##### Best Performing Regression Model
```
Model: Random Forest Regressor
Type: Tuned Model
Regression Metrics

MAE: 2.0688743398082728e-11
RMSE: 2.968035224756988e-11
R² Score: 1.0
Overall Score: 599985.2896639801
```

The tuned Random Forest Regressor achieved the best performance among all regression models and delivered highly accurate flight price predictions.

--- 
#### Classification Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score



#### 🏆 Best Performing Classification Model
```
Model: Logistic Regression
Type: Base Model
Classification Metrics

Accuracy: 98.88%
Precision: 100%
Recall: 97.77%
F1 Score: 98.87%
```

The Logistic Regression classifier achieved the highest overall classification performance and provided highly reliable gender prediction results.
The best-performing regression and classification models were tracked, versioned, and managed using MLflow.

---

## ⚙️ Technologies Used

#### Programming

- Python

#### Machine Learning

- Scikit-learn
- Pandas
- NumPy

#### Backend

- FastAPI
- Uvicorn

#### Frontend

- Streamlit
- Plotly

#### MLOps

- MLflow

#### Deployment

- Docker
- Docker Compose
- Kubernetes

---

## 🧠 System Architecture

```text
User
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
ML Pipeline
   ↓
MLflow Models
   ↓
Predictions & Recommendations
```

---

## ⚠️ IMPORTANT NOTE

### ❌ Local Hosting is NOT Supported

This project is NOT designed for full local execution.

Running both:

- Docker API
- Kubernetes Streamlit App

locally at the same time can create:

- Port conflicts
- API connection failures
- Docker networking issues
- Kubernetes communication issues

Therefore:

#### DO NOT try to fully run the project locally using normal localhost execution.

The project is designed to run using:

- Docker for API deployment
- Kubernetes for frontend deployment

This architecture is required for stable communication between the frontend and backend.

---

## 🐳 API Deployment (Docker)

The FastAPI backend is deployed using Docker because:

- MLflow model loading is heavy
- Large ML models increase startup time
- Kubernetes probes were timing out
- Docker handles ML workloads more reliably during development

---

## ☸️ Frontend Deployment (Kubernetes)

The Streamlit frontend app is deployed using Kubernetes.

The frontend connects to the Docker API using:

```text
http://host.docker.internal:8000
```

inside:

```text
kubernetes/app-deployment.yaml
```

---

## ⚠️ Kubernetes API Files Note

Inside the Kubernetes folder you will still find:

- api-deployment.yaml
- api-service.yaml

These files are currently NOT actively used.

Reason:

- API startup is heavy
- MLflow loading takes time
- Kubernetes health probes failed repeatedly
- API containers were crashing during startup

Future optimization can allow:

- Full Kubernetes deployment
- Faster startup
- Better caching
- Lightweight model serving
- Async model loading

---


## 📥 Download Complete Project

You can download the complete project files, pretrained models, MLflow logs, datasets, and deployment setup from the Google Drive link below:

https://drive.google.com/drive/folders/1tk4rF9QnBIy5TT8pIkp60Ic6Nj9ldqtB?usp=drive_link

The Google Drive folder contains:

- Complete source code
- Pretrained ML models
- MLflow database & logs
- Processed datasets
- Docker & Kubernetes files
- Notebook files
- Deployment configuration

Downloading the complete project folder is the recommended setup method. Then build the api and app image and deploy them - steps are given below.

---





## ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Predeep-Kumar/Voyage_Analytics-.git
cd Voyage_Analytics
```

---

## 2️⃣ Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🐳 Docker & Kubernetes Setup

Before running the project, make sure Docker Desktop is properly installed and configured on your system.

## 1️⃣ Install Docker Desktop

Download Docker Desktop based on your operating system and Sign up: - https://www.docker.com/products/docker-desktop/

- Windows
- macOS
- Linux

After installation:

- Open Docker Desktop
- Start the Docker Engine
- Wait until Docker status becomes active/running

---

## 2️⃣ Enable Kubernetes in Docker Desktop

Open:

```text
Docker Desktop → Settings → Kubernetes
```

Then:

- Enable Kubernetes
- Click "Apply & Restart"
- Wait for Kubernetes to fully start

You can verify Kubernetes is running using:

```bash
kubectl get nodes
```

If Kubernetes is working correctly, you should see:

```text
desktop-control-panel
```

in the node list.

---

## 3️⃣ Verify Docker Installation

Run:

```bash
docker --version
```

and:

```bash
docker ps
```

---

## 4️⃣ Verify Kubernetes Installation

Run:

```bash
kubectl version
```

---

## 🧠 Train Models & Log MLflow Models

NOTE:
If you downloaded the complete project folder from the provided Google Drive link, you can skip:

- Repository cloning
- Virtual environment setup
- Requirements installation
- MLflow model relogging

because the project already contains:

- Pretrained models
- MLflow logs & database
- Installed environment setup
- Project files and artifacts

You only need to run these steps if you are setting up the project from scratch or rebuilding the environment manually.

If MLflow errors occur, you can relog the models using:

```bash
docker exec -it voyage_api python mlops/run_pipeline.py
```

Otherwise, you can directly use the existing pretrained models and previous MLflow logs included in the project.

This step will:

- Train models
- Compare models
- Register best models
- Log MLflow experiments

---

## 🐳 Build API Docker Image And Deploying Using Docker

```bash
docker build -f Dockerfile.api -t voyage_analytics-api:v1 .
```

---

## ▶️ Run API using Docker Compose

```bash
docker compose up -d api
```

---

## 🔍 Check Running Containers

```bash
docker ps
```

You should see:

```text
voyage_api
```

running on port:

```text
8000
```

---

## ☸️ Building The App Image and Deploy Frontend App using Kubernetes

Build The App Image:
```
docker build --no-cache -f Dockerfile.app -t voyage_analytics-app:v2 .
```

Apply Kubernetes files:

```bash
kubectl apply -f kubernetes/app-deployment.yaml
```

Don't use this - ```kubectl apply -f kubernetes/``` it will apply all the kubernetes files including the api file too.

---

## 🔍 Check Kubernetes Pods Live

```bash
kubectl get pods -n voyage-analytics -w
```

It should be 1/1 Running.

---

## 🌐 Port Forward Streamlit App

```bash
kubectl port-forward service/voyage-app-service 8501:8501 -n voyage-analytics
```

Open:

```text
http://localhost:8501
```

---

## 📈 MLflow UI

Run:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

## 📊 Dashboard Features

The dashboard includes:

- Flight Price Prediction
- Gender Prediction
- Recommendation System
- Interactive Charts
- Recommendation Insights
- Model Monitoring
- System Health Status

---



## 📚 Learning Outcomes

This project provided hands-on experience in:

- Data Analytics
- Machine Learning
- Recommendation Systems
- API Development
- Docker
- Kubernetes
- MLflow
- Model Deployment
- Interactive Dashboard Design
- Production ML Workflows


---

## 💼 Business Use Cases

Voyage Analytics can be utilized across multiple real-world travel and business intelligence applications, helping organizations improve decision-making, customer experience, pricing strategies, and operational efficiency.

### Key Business Applications
- Airline ticket price forecasting and dynamic pricing optimization
- Personalized hotel recommendation systems for travelers
- Intelligent travel package generation combining flights and hotels
- Customer behavior and travel trend analysis
- AI-driven recommendation systems for travel platforms
- Revenue optimization for airlines and travel agencies
- Budget-aware destination and hotel recommendations
- Demand forecasting using machine learning models
- Data-driven travel business analytics and reporting
- Scalable deployment of ML applications using Docker and Kubernetes
- End-to-end MLOps workflow implementation using MLflow
- Production-ready AI system architecture for real-world deployment

---

## 🏁 Conclusion

Voyage Analytics demonstrates a complete end-to-end AI and MLOps ecosystem integrating machine learning, recommendation systems, scalable deployment, and modern UI/UX design. The project showcases how intelligent travel platforms can leverage predictive analytics, recommendation engines, Docker, Kubernetes, FastAPI, Streamlit, and MLflow to build scalable and production-ready AI solutions.

---

## 👨‍💻 Authors


👨‍💻 Project Team
- Predeep Kumar
- Shivika Verma
- Abhishek R
- Akanksha Rani

Built with ❤️ as an end-to-end AI-powered travel analytics platform using Machine Learning, Recommendation Systems, MLOps, Docker, and Kubernetes.
