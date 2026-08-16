# Heart Disease Prediction Using Data Science and Machine Learning

## 1. Project Overview
This project builds an end-to-end beginner-friendly data science and machine learning workflow to predict whether a patient is likely to have heart disease. The project uses the UCI Heart Disease dataset from the UC Irvine Machine Learning Repository.

This is an educational academic project and is not a medical diagnosis.

## 2. Problem Statement
Heart disease is a major health problem worldwide. Early identification can help healthcare professionals understand risk patterns and guide decisions. In this project, we use patient features to train a machine learning model that classifies patients into:

- 0 = No heart disease
- 1 = Heart disease present

## 3. Objectives
- Collect and understand the dataset
- Clean and prepare the data
- Perform exploratory data analysis
- Preprocess features without data leakage
- Train and compare multiple machine learning models
- Evaluate model performance using appropriate metrics
- Tune a promising model
- Save the final model for future use
- Create a simple prediction interface

## 4. Dataset
The dataset is obtained from the UCI Machine Learning Repository using the `ucimlrepo` package.

Source: UCI Heart Disease dataset (ID: 45)

The dataset includes patient health features such as age, sex, cholesterol, blood pressure, and exercise-induced angina.

## 5. Features
The main features used in the project are:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

The target variable is the heart disease class label.

## 6. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- UCI ML Repo
- Joblib
- Streamlit

## 7. Project Structure

```text
heart-disease-prediction/
├── data/
│   └── heart_disease.csv
├── notebooks/
│   ├── 01_data_collection_and_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_data_preprocessing.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── models/
│   └── heart_disease_model.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── app.py
├── visualizations/
├── requirements.txt
├── README.md
├── LEARNING_GUIDE.md
├── .gitignore
└── venv/
```

## 8. Data Preprocessing
The preprocessing step includes:

- loading the data
- converting the multi-class target to a binary target
- handling missing values using median and most frequent imputation
- scaling numeric features
- one-hot encoding categorical features
- preventing leakage by fitting preprocessing on the training set only

## 9. Exploratory Data Analysis
The project contains visualizations for:

- target distribution
- age distribution
- cholesterol distribution
- resting blood pressure distribution
- maximum heart rate distribution
- heart disease vs age
- heart disease vs sex
- heart disease vs chest pain type
- heart disease vs cholesterol
- heart disease vs maximum heart rate
- heart disease vs exercise-induced angina
- correlation heatmap

## 10. Machine Learning Algorithms
The project compares the following models:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

## 11. Evaluation Metrics
The project evaluates each model with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve

## 12. Results
Model results will be generated during training and evaluation. The final chosen model is selected after comparing cross-validation performance, test performance, and overall stability.

## 13. How to Install
```bash
cd heart-disease-prediction
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 14. How to Run
```bash
cd heart-disease-prediction
python src/train.py
python src/predict.py --age 52 --sex 1 --cp 0 --trestbps 125 --chol 212 --fbs 0 --restecg 1 --thalach 168 --exang 0 --oldpeak 1.0 --slope 2 --ca 0 --thal 3
streamlit run app/app.py
```

## 15. How to Use the Prediction Application
Open the Streamlit app in a browser and enter patient values. The app will show:

- model prediction
- probability of heart disease
- educational disclaimer

## 16. Limitations
- The model is meant for study and education, not diagnosis
- Data comes from a single hospital/clinical repository
- Some features may require domain expert validation
- Real-world deployment would require broader dataset and clinical review

## 17. Future Scope
- Add more feature engineering
- Compare more algorithms
- Use better interpretability tools
- Build a larger real-world clinical dataset workflow

## 18. Disclaimer
This project is intended for educational and academic learning. It does not diagnose disease or replace medical advice.
