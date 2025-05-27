# Group_8_loan_default_prediction
# 🏦 Loan Default Prediction using Machine Learning

## 📌 Project Objective
The aim of this project is to **predict whether a loan applicant will default** (Loan_Status: Y or N) using machine learning classification models. We apply **Decision Tree** and **Random Forest** algorithms to make predictions based on applicant financial and demographic features. The project also includes **feature importance analysis** and **data visualizations** for deeper insights.

---

## 📁 Dataset

The dataset consists of two files:

- `train_u6lujuX_CVtuZ9i.csv` — contains labeled training data.
- `test_Y3wMUE5_7gLdaTN.csv` — contains unlabeled test data for prediction.

Each record contains details like:
- Gender, Married, Dependents, Education
- Self_Employed, ApplicantIncome, CoapplicantIncome
- LoanAmount, Loan_Amount_Term, Credit_History
- Property_Area, and Loan_Status

---

## ⚙️ Technologies Used

| Tool/Library        | Purpose                                  |
|---------------------|------------------------------------------|
| Python              | Programming language                     |
| Pandas              | Data manipulation and preprocessing      |
| NumPy               | Numerical operations                     |
| Matplotlib / Seaborn| Data visualization                       |
| Scikit-learn        | Machine learning models and evaluation   |

---

## 📊 Exploratory Data Analysis

The code includes the following visualizations:
- Distribution of Loan Status
- Applicant Income Histogram
- Loan Amount vs Loan Status (Boxplot)
- Correlation Heatmap of all features
- Feature Importance (Random Forest)

These help understand patterns in the data and the key drivers of loan defaults.

---

## 🧠 Machine Learning Models

### 1. **Decision Tree Classifier**
- A tree-based model that splits data based on conditions.
- Easy to interpret but may overfit.

### 2. **Random Forest Classifier**
- An ensemble method using multiple decision trees.
- More robust and accurate than a single decision tree.
- Provides feature importance scores.

---

## 📈 Model Evaluation

Models are evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)

The Random Forest Classifier typically outperforms the Decision Tree by reducing overfitting and improving generalization.

---

## 🧪 Workflow Summary

1. Load and combine datasets.
2. Handle missing values.
3. Encode categorical variables using `LabelEncoder`.
4. Split data into training and validation sets.
5. Train Decision Tree and Random Forest classifiers.
6. Evaluate models on validation data.
7. Generate data visualizations.
8. Analyze feature importance.

---

## 📂 File Structure

├── train_u6lujuX_CVtuZ9i.csv # Training dataset
├── test_Y3wMUE5_7gLdaTN.csv # Test dataset
├── loan_default_prediction.py # Main Python script
├── README.md # Project overview


---

## ✅ Results

- **Random Forest Accuracy**: ~78–82% (depending on train/test split)
- **Key Features**: Credit History, Loan Amount, Applicant Income, Property Area

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Cross-validation for better performance estimates
- Model explainability with SHAP or LIME
- Deployment using Flask or Streamlit

---

## 📬 Contact

For questions or collaboration, feel free to reach out via GitHub or email.

---

![Screenshot 2025-05-27 111709](https://github.com/user-attachments/assets/bd6f89a7-e4a1-4152-82ae-f74f42a0809d)
![Screenshot 2025-05-27 112012](https://github.com/user-attachments/assets/24b9af8d-3bf8-40b4-890b-e0bbe35381f3)
![Screenshot 2025-05-27 111951](https://github.com/user-attachments/assets/18eb45ff-5b38-4f5e-a984-48bf84c9b642)
![Screenshot 2025-05-27 111728](https://github.com/user-attachments/assets/526ece87-cbee-492b-a9f0-69ca85a46d38)

