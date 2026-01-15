import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("📊 Machine Learning Classification App")

# -------------------------
# Dataset Upload (REQUIRED)
# -------------------------
st.header("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Basic Preprocessing
    # -------------------------
    st.header("🧹 Data Preprocessing")

    if 'Survived' not in df.columns:
        st.error("Target column 'Survived' not found in dataset.")
        st.stop()

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    if 'Cabin' in df.columns:
        df['Cabin_Known'] = df['Cabin'].notnull().astype(int)
        df.drop('Cabin', axis=1, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # -------------------------
    # Feature & Target
    # -------------------------
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # Model Selection (REQUIRED)
    # -------------------------
    st.header("🤖 Model Selection")

    model_name = st.selectbox(
        "Choose a Classification Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest"
        )
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    model = models[model_name]

    if st.button("Train and Evaluate Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -------------------------
        # Metrics Display (REQUIRED)
        # -------------------------
        st.header("📈 Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))

        # -------------------------
        # Confusion Matrix (REQUIRED)
        # -------------------------
        st.header("🧩 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------------
        # Classification Report
        # -------------------------
        st.header("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))
