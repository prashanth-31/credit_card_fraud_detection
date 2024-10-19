import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('creditcard.csv')  # Adjust with your dataset path
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target (fraud/not fraud)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separate the classes
X_class_0 = X[y == 0]
X_class_1 = X[y == 1]

# Split class 1 into train and test (2/3 for training, 1/3 for testing)
n_class_1_train = int(len(X_class_1) * (2 / 3))

X_train_class_1 = X_class_1[:n_class_1_train]
X_test_class_1 = X_class_1[n_class_1_train:]

# Split class 0 into train and test (stratify based on the remaining class proportions)
X_train_class_0, X_test_class_0, y_train_class_0, y_test_class_0 = train_test_split(
    X_class_0, y[y == 0], test_size=len(X_test_class_1), random_state=42, stratify=y_class_0)

# Combine the classes back into a single dataset
X_train = np.vstack((X_train_class_0, X_train_class_1))
y_train = np.concatenate((y_train_class_0, np.ones(len(X_train_class_1))))

X_test = np.vstack((X_test_class_0, X_test_class_1))
y_test = np.concatenate((y_test_class_0, np.ones(len(X_test_class_1))))

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build a neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = build_model()
class_weight = {0: 1, 1: 5}  # Weight for fraud cases
model.fit(X_train_resampled, y_train_resampled, epochs=3, batch_size=32, class_weight=class_weight, validation_split=0.2)

# Function to calculate model performance
def get_model_performance(model, X, y, threshold=0.5):
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > threshold).astype("int32")
    return (accuracy_score(y, y_pred),
            precision_score(y, y_pred, zero_division=0),
            recall_score(y, y_pred, zero_division=0),
            f1_score(y, y_pred, zero_division=0),
            y_pred)

# Generate adversarial examples for testing
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)
    return np.clip(X + noise, 0, None)

X_adv_test = generate_adversarial_examples(X_test)

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train_resampled[:100])  # Limit for faster SHAP calculations

# Streamlit app
st.title("Fraud Detection Model Dashboard")

# Sidebar navigation
section = st.sidebar.radio("Go to", ["Model Overview", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Performance on clean data
    clean_metrics = get_model_performance(model, X_test, y_test)
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_metrics[0]:.4f}")
    st.write(f"Precision: {clean_metrics[1]:.4f}")
    st.write(f"Recall: {clean_metrics[2]:.4f}")
    st.write(f"F1-Score: {clean_metrics[3]:.4f}")
    
    # Confusion matrix for clean data
    st.subheader("Confusion Matrix for Clean Data")
    sns.heatmap(confusion_matrix(y_test, clean_metrics[4]), annot=True, fmt="d", cmap="Blues")
    st.pyplot()

# Explainability Section
elif section == "Explainability":
    st.header("Explainability with SHAP")
    
    # Feature importance plot
    st.subheader("Feature Importance Plot (SHAP)")
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot()

# Interactive Prediction Tool Section
elif section == "Interactive Prediction Tool":
    st.header("Interactive Prediction Tool")
    
    # Input features for a new transaction
    st.subheader("Input Transaction Features")
    transaction_input = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(X.shape[1])]
    
    # Predict fraud/not fraud
    transaction_input_scaled = scaler.transform([transaction_input])
    prediction_prob = model.predict(transaction_input_scaled)
    prediction = "Fraud" if prediction_prob[0][0] > 0.5 else "Not Fraud"

    st.subheader("Prediction Result")
    st.write(f"Prediction Probability: {prediction_prob[0][0]:.4f}")
    st.write(f"Prediction: {prediction}")
