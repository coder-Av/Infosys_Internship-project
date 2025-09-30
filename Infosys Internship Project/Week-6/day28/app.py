import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_model():
    """Loads the pre-trained model from disk."""
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    """Loads and prepares the Iris dataset."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    X['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return X, iris.target_names, iris.feature_names

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("🌸 Iris Flower Classification App")
st.write("An interactive web app for predicting Iris species using a trained ML model.")

# Sidebar - mode selection
mode = st.sidebar.radio("Choose mode:", ("Prediction", "Data Exploration"))

if mode == "Prediction":
    model = load_model()
    _, target_names, feature_names = load_data()
    
    st.header("🔮 Make a Prediction")

    # Input sliders
    # Use hardcoded reasonable defaults for sliders for a cleaner look
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
    petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.subheader("Prediction Result")
    st.success(f"**Predicted Species:** {target_names[prediction][0]}")

    st.subheader("Prediction Probabilities")
    st.write(dict(zip(target_names, prediction_proba[0])))

elif mode == "Data Exploration":
    st.header("📊 Dataset Exploration")
    st.write("Explore the Iris dataset with simple visualizations.")

    # Histogram
    X, _, feature_names = load_data()
    feature = st.selectbox("Select feature for histogram:", feature_names)
    fig, ax = plt.subplots()
    sns.histplot(data=X, x=feature, hue="species", kde=True, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")
    st.subheader("Feature Relationship Scatter Plot")
    # Scatter plot
    feature_x = st.selectbox("X-axis:", feature_names, index=0)
    feature_y = st.selectbox("Y-axis:", feature_names, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(data=X, x=feature_x, y=feature_y, hue="species", ax=ax, alpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)
