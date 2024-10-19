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
from lime import lime_tabular

# Load and preprocess dataset
data = pd.read_csv('creditcard.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train_class_0, X_test_class_0, y_train_class_0, y_test_class_0 = train_test_split(
    X[y == 0], y[y == 0], test_size=0.5, random_state=42)
X_train_class_1, X_test_class_1, y_train_class_1, y_test_class_1 = train_test_split(
    X[y == 1], y[y == 1], test_size=0.5, random_state=42)

X_train = np.vstack((X_train_class_0, X_train_class_1))
y_train = np.concatenate((y_train_class_0, y_train_class_1))
X_test = np.vstack((X_test_class_0, X_test_class_1))
y_test = np.concatenate((y_test_class_0, y_test_class_1))

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build and train the neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
class_weight = {0: 1, 1: 5}

# Generate adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)
    return np.clip(X + noise, 0, None)

X_adv = generate_adversarial_examples(X_train_resampled)

# Train the model with original and adversarial examples
X_combined = np.vstack((X_train_resampled, X_adv))
y_combined = np.concatenate((y_train_resampled, y_train_resampled))
model.fit(X_combined, y_combined, epochs=2, batch_size=32, class_weight=class_weight, validation_split=0.2)

# Model performance evaluation
def get_model_performance(model, X, y, threshold=0.5):
    y_pred = (model.predict(X) > threshold).astype("int32")
    return (accuracy_score(y, y_pred), precision_score(y, y_pred, zero_division=0), 
            recall_score(y, y_pred, zero_division=0), f1_score(y, y_pred, zero_division=0), y_pred)

# Generate adversarial examples for testing
X_adv_test = generate_adversarial_examples(X_test)

# Main Streamlit app
st.title("Fraud Detection Model Dashboard")

section = st.sidebar.radio("Go to", ["Model Overview", "Explainability", "Interactive Prediction Tool"])

if section == "Model Overview":
    clean_metrics = get_model_performance(model, X_test, y_test)
    adv_metrics = get_model_performance(model, X_adv_test, y_test)
    
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_metrics[0]:.4f}, Precision: {clean_metrics[1]:.4f}, Recall: {clean_metrics[2]:.4f}, F1-Score: {clean_metrics[3]:.4f}")
    
    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_metrics[0]:.4f}, Precision: {adv_metrics[1]:.4f}, Recall: {adv_metrics[2]:.4f}, F1-Score: {adv_metrics[3]:.4f}")
    
    # Confusion Matrices
    for title, y_pred in [("Clean Data", clean_metrics[4]), ("Adversarial Data", adv_metrics[4])]:
        st.subheader(f"Confusion Matrix for {title}")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot()

elif section == "Explainability":
    st.header("Explainability with LIME")
    idx = st.slider("Select Transaction Index", 0, len(X_test) - 1)
    exp = lime_tabular.LimeTabularExplainer(
        training_data=X_train_resampled,
        feature_names=[f"Feature {i+1}" for i in range(X_train_resampled.shape[1])],
        class_names=["Not Fraud", "Fraud"],
        mode='classification').explain_instance(X_test[idx], model.predict, num_features=10)
    
    st.write("LIME Explanation for the Selected Transaction:")
    exp.as_pyplot_figure()
    st.pyplot()

elif section == "Interactive Prediction Tool":
    st.header("Interactive Prediction Tool")
    transaction_input = [st.number_input(f"Feature {i+1}", value=float(X_test[0, i]), step=0.01) for i in range(X_test.shape[1])]
    transaction_input = np.array(transaction_input).reshape(1, -1)
    transaction_input_scaled = scaler.transform(transaction_input)
    
    prediction_prob = model.predict(transaction_input_scaled)
    prediction = "Fraud" if prediction_prob[0][0] > 0.5 else "Not Fraud"
    
    st.subheader("Prediction Result")
    st.write(f"Prediction Probability: {prediction_prob[0][0]:.4f}, Prediction: {prediction}")
    
    if st.button("Explain Prediction with LIME"):
        exp = lime_tabular.LimeTabularExplainer(
            training_data=X_train_resampled,
            feature_names=[f"Feature {i+1}" for i in range(X_train_resampled.shape[1])],
            class_names=["Not Fraud", "Fraud"],
            mode='classification').explain_instance(transaction_input_scaled[0], model.predict)
        st.subheader("LIME Explanation")
        exp.as_pyplot_figure()
        st.pyplot()
