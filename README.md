# 🧠 Autism Prediction using Machine Learning

This project uses machine learning to predict the likelihood of Autism Spectrum Disorder (ASD) based on personal and behavioral survey data. The workflow includes data preprocessing, visualization, handling class imbalance using SMOTE, model training, and hyperparameter tuning using RandomizedSearchCV.

---

## 📂 Dataset

- **Source**: `train.csv`
- **Features**: Demographic info, 10 Q&A autism screening scores, and other attributes
- **Target Variable**: `Class/ASD` (Yes = 1, No = 0)

---

## ✅ Project Workflow

### 1. **Data Preprocessing**
- Cleaned inconsistent country names
- Handled missing and noisy values
- Encoded categorical features using `LabelEncoder`
- Replaced outliers in `age` and `result` columns using the **IQR method**

### 2. **Exploratory Data Analysis (EDA)**
- Plotted histograms and boxplots for numerical features
- Countplots for categorical variables
- Visualized correlations using a **heatmap**

### 3. **Handling Class Imbalance**
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the target class

### 4. **Model Training**
Trained three models:
- Decision Tree
- Random Forest
- XGBoost

Used **5-fold cross-validation** to evaluate performance.

### 5. **Hyperparameter Tuning**
Used **RandomizedSearchCV** to optimize model parameters:
- 20 iterations for each model
- Best model selected based on cross-validation accuracy

### 6. **Model Evaluation**
Evaluated the best model on the test set using:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 7. **Model Saving**
Saved the best model and encoders using `pickle` for future predictions.

---

## 📊 Results

- **Best Model**: Selected from Decision Tree, Random Forest, or XGBoost (based on accuracy)
- **Performance**: Accuracy and classification report provided for test set evaluation

---

## 🧪 Libraries Used

- `pandas`, `numpy`, `seaborn`, `matplotlib`
- `scikit-learn` (`LabelEncoder`, `DecisionTreeClassifier`, `RandomForestClassifier`, `RandomizedSearchCV`, `cross_val_score`)
- `xgboost`
- `imblearn` (`SMOTE`)
- `pickle`

---

## 💾 How to Use

1. Load dataset `train.csv`
2. Run `Autism Prediction.ipynb`
3. The trained model will be saved as `best_model.pkl`
4. You can use this model to predict autism likelihood on new data

---

## 📌 Author

M Zohaib Shahid  
_Data Science Enthusiast | ML Developer_  
📫 [LinkedIn Profile](https://www.linkedin.com) *(update with your link)*

---

