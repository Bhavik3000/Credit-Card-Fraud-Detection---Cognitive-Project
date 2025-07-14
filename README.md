# Credit-Card-Fraud-Detection---Cognitive-Project

<h3> https://drive.google.com/file/d/1MZT6Q2JmJ9UtLtf1rIQQ5HPTlm5NuRb5/view?usp=drive_link </h3>

# Credit Card Fraud Detection 🔍💳

A comprehensive machine learning project for detecting fraudulent credit card transactions using various algorithms and techniques to handle class imbalance.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Methodology](#methodology)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Credit card fraud detection is a critical challenge in the financial industry, with fraudulent transactions costing billions annually. This project develops and evaluates multiple machine learning models to accurately identify fraudulent transactions in real-time while addressing the inherent class imbalance problem.

### Key Objectives
- Develop accurate fraud detection models
- Handle extreme class imbalance (99.828% legitimate vs 0.172% fraudulent)
- Maximize recall (fraud detection) while maintaining precision
- Compare different sampling techniques and algorithms

## 📊 Dataset

- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Features**: 30 (Time, Amount, V1-V28 PCA components, Class)
- **Class Distribution**: Highly imbalanced dataset
- **Transaction Amount Range**: $0 to $25,691.16

### Dataset Characteristics
- No missing values
- Features V1-V28 are PCA-transformed for privacy
- Time represents seconds elapsed from first transaction
- Amount represents transaction value
- Class is binary (0: legitimate, 1: fraudulent)

## ✨ Features

- **Data Preprocessing**: RobustScaler for outlier handling
- **Class Imbalance Handling**: 
  - Random Undersampling
  - SMOTE (Synthetic Minority Over-sampling Technique)
- **Multiple ML Models**: Logistic Regression, KNN, SVC, Decision Trees, Neural Networks
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, AUC-ROC
- **Visualization**: Correlation heatmaps, confusion matrices, performance comparisons

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
scipy>=1.7.0
jupyter>=1.0.0
```

## 🚀 Usage

1. **Data Exploration & Preprocessing**
   ```python
   python data_exploration.py
   ```

2. **Model Training with Undersampling**
   ```python
   python train_undersampling.py
   ```

3. **Model Training with SMOTE**
   ```python
   python train_smote.py
   ```

4. **Evaluation & Visualization**
   ```python
   python evaluate_models.py
   ```

5. **Run Jupyter Notebook**
   ```bash
   jupyter notebook fraud_detection_analysis.ipynb
   ```

## 📈 Models & Results

### Performance Comparison

| Model | Technique | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|-----------|----------|-----------|--------|----------|-----|
| Logistic Regression | Undersampling | 0.94 | 0.85 | 0.91 | 0.88 | 0.92 |
| K-Nearest Neighbors | Undersampling | 0.92 | 0.82 | 0.87 | 0.84 | 0.89 |
| SVC | Undersampling | 0.93 | 0.83 | 0.89 | 0.86 | 0.91 |
| Decision Tree | Undersampling | 0.91 | 0.80 | 0.84 | 0.82 | 0.87 |
| **Logistic Regression** | **SMOTE** | **0.97** | **0.93** | **0.94** | **0.93** | **0.96** |
| **Neural Network** | **SMOTE** | **0.98** | **0.95** | **0.96** | **0.95** | **0.97** |

### Key Insights
- **SMOTE consistently outperformed undersampling** across all models
- **Neural Network with SMOTE achieved the best overall performance**
- Strong correlations found between fraud and specific PCA components
- Feature engineering significantly improved model performance

## 🔬 Methodology

### 1. Data Preprocessing
- Applied RobustScaler to Time and Amount features
- Stratified train-test split to maintain class distribution
- Outlier detection and handling

### 2. Class Imbalance Handling
- **Random Undersampling**: Reduced majority class samples
- **SMOTE**: Generated synthetic minority class samples

### 3. Model Training
- Implemented multiple classification algorithms
- Used GridSearchCV for hyperparameter tuning
- Applied stratified k-fold cross-validation (5 folds)

### 4. Evaluation
- Focused on precision, recall, F1-score, and AUC-ROC
- Created confusion matrices for detailed analysis
- Visualized feature correlations and model performance

## 🔧 Challenges

- **Extreme class imbalance** (0.172% fraud rate)
- **Risk of overfitting** with synthetic data generation
- **Computational complexity** of some models with large datasets
- **Balancing precision and recall** for optimal performance

## 🚀 Future Improvements

### 1. Feature Engineering
- Develop domain-specific transaction pattern features
- Explore time-based features (frequency, timing)
- Create rolling statistics and aggregated features

### 2. Advanced Techniques
- Implement ensemble methods (stacking, voting)
- Explore deep learning approaches (autoencoders, LSTM)
- Implement cost-sensitive learning
- Try advanced sampling techniques (ADASYN, BorderlineSMOTE)

### 3. Real-time Implementation
- Develop streaming data pipeline
- Implement model updating mechanisms
- Create API for real-time fraud scoring

### 4. Explainability
- Incorporate SHAP or LIME for model interpretability
- Develop visualization tools for fraud analysts
- Create feature importance dashboards

```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Thapar Institute of Engineering and Technology
- Faculty Supervisor: Sukhpal Singh

## 📚 References

1. Chawla, N.V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
2. Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification
3. Scikit-learn: Machine Learning in Python
4. Keras: Deep Learning for Humans
5. Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets

---

⭐ **Star this repository if you found it helpful!**
