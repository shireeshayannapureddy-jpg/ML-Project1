import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Analysis & Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
st.header("📂 Load Dataset")

df = pd.read_csv("Titanic-Dataset.csv")
st.dataframe(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
st.header("🧹 Data Cleaning")

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin_Known'] = df['Cabin'].notnull().astype(int)
df.drop('Cabin', axis=1, inplace=True)

st.write("Missing values handled successfully.")

# -----------------------------
# Encode Categorical Variables
# -----------------------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -----------------------------
# EDA Section
# -----------------------------
st.header("📊 Exploratory Data Analysis")

fig1, ax1 = plt.subplots()
sns.countplot(x='Cabin_Known', hue='Survived', data=df, ax=ax1)
ax1.set_title("Survival vs Cabin Availability")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0,12,20,40,60,80],
    labels=['Child','Teen','Adult','Middle-Aged','Senior']
)
sns.countplot(x='Age_Group', hue='Survived', data=df, ax=ax2)
ax2.set_title("Survival by Age Group")
st.pyplot(fig2)

# -----------------------------
# Feature Selection
# -----------------------------
selected_features = [
    'Sex', 'Pclass', 'Fare',
    'Cabin_Known', 'Age',
    'Embarked', 'Survived'
]

df_final = df[selected_features]
X = df_final.drop('Survived', axis=1)
y = df_final['Survived']

# -----------------------------
# Train-Test Split & Scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# -----------------------------
# Model Training & Evaluation
# -----------------------------
st.header("🤖 Model Comparison")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "NA",
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)
