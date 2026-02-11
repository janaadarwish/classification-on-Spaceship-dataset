# Spaceship Titanic – Machine Learning Classification

## Project Overview

This project predicts whether a passenger was transported to another dimension during the Spaceship Titanic incident.

Using historical passenger data, multiple machine learning models were built and evaluated to classify passengers as:

- **Transported (True)**
- **Not Transported (False)**

This is a **supervised binary classification problem**.

 Dataset Source:  
https://www.kaggle.com/competitions/spaceship-titanic

---

## Project Workflow

### Exploratory Data Analysis (EDA)
- Explored dataset structure and feature types
- Identified numerical and categorical columns
- Analyzed missing values
- Created visualizations to understand feature relationships

### Data Cleaning
- Filled missing values (median for numerical, mode for categorical)
- Removed duplicate records
- Dropped irrelevant columns
- Checked and handled outliers

### Feature Engineering
- Created new features such as **Total Spending**
- Extracted useful information from existing columns

### Encoding & Scaling
- Applied One-Hot Encoding to categorical variables
- Scaled numerical features using `StandardScaler`

### Data Splitting
- Split data into training and testing sets
- Used cross-validation for model tuning

---

## Models Implemented

### Logistic Regression
A baseline linear classification model.

### K-Nearest Neighbors (KNN)
- Used Cross-Validation to select the best value of K.
- Trained final model using the selected K.

### Neural Network (Keras)
- Input layer
- At least one hidden layer
- Output layer with **sigmoid activation**
- Tuned hyperparameters (neurons, learning rate, batch size, epochs)

---

## Model Evaluation

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- Confusion Matrix

Training and testing performance were compared to determine:
- Overfitting
- Underfitting
- Right-fitting

---

## Final Model Comparison

| Model | Training Accuracy | Testing Accuracy | Fit Status |
|-------|-------------------|-----------------|------------|
| Logistic Regression |  |  |  |
| KNN |  |  |  |
| Neural Network |  |  |  |

*(Insert your actual results here.)*

---

## Unsupervised Learning Extension (Bonus)

- Removed target column
- Selected meaningful numerical features
- Applied K-Means Clustering
- Visualized passenger clusters

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
