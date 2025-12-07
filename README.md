# 🚕 Uber Demand Prediction in New York City

A machine learning project that predicts taxi demand across 30 regions in New York City using historical trip data. The project includes a complete MLOps pipeline with data processing, feature engineering, model training, and an interactive Streamlit web application for real-time demand visualization.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.39.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Data Pipeline](#data-pipeline)
- [Streamlit App](#streamlit-app)
- [MLOps Integration](#mlops-integration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project predicts taxi demand for the next 15-minute interval across 30 distinct regions in New York City. The solution uses:

- **Clustering**: MiniBatch K-Means to divide NYC into 30 regions based on pickup locations
- **Time Series Features**: Lag features and temporal patterns (day of week, month)
- **Smoothing**: Exponential Weighted Moving Average (EWMA) for demand smoothing
- **Regression Model**: Linear Regression for demand prediction
- **Interactive Dashboard**: Streamlit app for real-time demand visualization

## ✨ Features

- 🗺️ **Geographic Clustering**: Automatically divides NYC into 30 regions using K-Means clustering
- 📊 **Time Series Analysis**: Captures temporal patterns with lag features and datetime features
- 🎨 **Interactive Dashboard**: Beautiful Streamlit app with map visualization
- 🔄 **MLOps Pipeline**: Complete DVC pipeline for reproducible experiments
- 📈 **Model Tracking**: MLflow integration via DagsHub for experiment tracking
- 🚀 **Production Ready**: Trained models ready for deployment
- 📱 **Real-time Predictions**: Predict demand for any date/time in March 2016

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models and preprocessing
- **Streamlit** - Interactive web application

### MLOps & Data Versioning
- **DVC** - Data version control and pipeline management
- **MLflow** - Experiment tracking and model registry
- **DagsHub** - MLflow tracking server

### Data Processing
- **Dask** - Parallel computing for large datasets
- **Joblib** - Model serialization

## 📁 Project Structure

```
uber-demand-price-prediction/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── params.yaml                 # Model hyperparameters
├── dvc.yaml                    # DVC pipeline configuration
├── Makefile                    # Make commands for common tasks
│
├── data/
│   ├── raw/                    # Original, immutable data
│   │   ├── yellow_tripdata_2016-01.csv
│   │   ├── yellow_tripdata_2016-02.csv
│   │   └── yellow_tripdata_2016-03.csv
│   ├── interim/                # Intermediate data
│   │   └── df_without_outliers.csv
│   └── processed/               # Final datasets
│       ├── train.csv
│       ├── test.csv
│       └── resampled_data.csv
│
├── models/                     # Trained models
│   ├── model.joblib            # Linear Regression model
│   ├── encoder.joblib          # Feature encoder
│   ├── scaler.joblib           # StandardScaler
│   └── mb_kmeans.joblib        # K-Means clustering model
│
├── notebooks/                   # Jupyter notebooks for EDA
│   ├── EDA-Demand-Prediction.ipynb
│   ├── Breaking_NYC_to_Regions.ipynb
│   ├── Creating-Historical-Data.ipynb
│   ├── Training-Baseline-Model.ipynb
│   └── ...
│
├── src/
│   ├── data/
│   │   └── data_ingestion.py   # Data loading and preprocessing
│   ├── features/
│   │   ├── extract_features.py # Feature extraction & clustering
│   │   └── feature_processing.py # Lag features & time series
│   └── models/
│       ├── train.py            # Model training
│       ├── evaluate.py         # Model evaluation
│       └── register_model.py   # MLflow model registration
│
└── docs/                       # Documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) DVC for pipeline execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmitZala/uber-demand-price-prediction.git
cd uber-demand-price-prediction
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda create -n uber-demand python=3.10
conda activate uber-demand
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Install DVC

If you want to run the full pipeline:

```bash
pip install dvc
```

## 💻 Usage

### Running the Streamlit App

The easiest way to use the project is through the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features of the App:**
- Select date and time (March 2016)
- View your current location on the map
- See demand predictions for all 30 regions
- Filter to view only neighborhood regions
- Interactive map with color-coded regions

### Running the Full Pipeline

If you want to retrain the models from scratch:

```bash
# Install DVC first
pip install dvc

# Run the complete pipeline
dvc repro
```

This will execute all stages:
1. **Data Ingestion**: Load and clean raw data
2. **Extract Features**: Create regions using K-Means, apply EWMA smoothing
3. **Feature Processing**: Generate lag features and split train/test
4. **Train**: Train the Linear Regression model
5. **Evaluate**: Evaluate model performance
6. **Register Model**: Register model in MLflow

### Individual Pipeline Stages

You can also run individual stages:

```bash
# Data ingestion
python src/data/data_ingestion.py

# Feature extraction
python src/features/extract_features.py

# Feature processing
python src/features/feature_processing.py

# Model training
python src/models/train.py

# Model evaluation
python src/models/evaluate.py
```

## 🤖 Model Details

### Architecture

1. **Geographic Clustering**
   - Method: MiniBatch K-Means
   - Number of clusters: 30 regions
   - Features: Scaled pickup coordinates (longitude, latitude)
   - Purpose: Divide NYC into distinct demand regions

2. **Feature Engineering**
   - **Lag Features**: Previous 1-4 time intervals (15-min windows)
   - **Temporal Features**: Day of week, month
   - **Smoothing**: EWMA with alpha=0.4 for average pickups
   - **Encoding**: One-hot encoding for categorical features

3. **Model**
   - Algorithm: Linear Regression
   - Input: Lag features + temporal features + region + day_of_week
   - Output: Predicted number of pickups for next 15-minute interval

### Hyperparameters

See `params.yaml`:
```yaml
extract_features:
  mini_batch_kmeans:
    n_clusters: 30
    n_init: 10
    random_state: 42
  ewma:
    alpha: 0.4
```

## 🔄 Data Pipeline

The project uses DVC for pipeline orchestration:

```
Raw Data → Data Ingestion → Feature Extraction → Feature Processing → Training → Evaluation
```

### Pipeline Stages

1. **data_ingestion**: Cleans raw taxi data, removes outliers
2. **extract_features**: Creates regions, applies EWMA smoothing
3. **feature_processing**: Generates lag features, splits data
4. **train**: Trains Linear Regression model
5. **evaluate**: Evaluates model on test set
6. **register_model**: Registers model in MLflow

## 🎨 Streamlit App

The interactive dashboard (`app.py`) provides:

- **Date/Time Selection**: Choose any date/time in March 2016
- **Location Sampling**: Randomly samples a location from NYC
- **Map Visualization**: 
  - Complete NYC map with all 30 regions
  - Neighborhood view showing 9 nearest regions
- **Demand Predictions**: Real-time predictions for each region
- **Color-coded Regions**: Visual representation of demand levels

### App Features

- Interactive map with Streamlit's native map component
- Real-time demand predictions
- Region-based filtering
- Beautiful UI with progress indicators

## 📊 MLOps Integration

### MLflow Tracking

The project integrates with MLflow via DagsHub:

- **Tracking URI**: `https://dagshub.com/AmitZala/uber-demand-price-prediction.mlflow`
- **Model Registry**: Models are registered with versioning
- **Experiment Tracking**: All runs are logged with metrics

### DVC Pipeline

- **Reproducibility**: All pipeline stages are versioned
- **Dependency Tracking**: Automatic dependency resolution
- **Parameter Management**: Hyperparameters in `params.yaml`

## 📈 Results

Model performance metrics are saved in `run_information.json` after evaluation. The model predicts demand for 30 regions across NYC with temporal patterns captured through lag features.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Amit Zala**

- GitHub: [@AmitZala](https://github.com/AmitZala)
- DagsHub: [uber-demand-price-prediction](https://dagshub.com/AmitZala/uber-demand-price-prediction)

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- CookieCutter Data Science project template
- Streamlit for the amazing framework
- DagsHub for MLflow hosting

---

⭐ If you find this project helpful, please give it a star!
