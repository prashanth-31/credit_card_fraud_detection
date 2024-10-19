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

# Separate the dataset into two parts based on class
X_class_0 = X[y == 0]
y_class_0 = y[y == 0]
X_class_1 = X[y == 1]
y_class_1 = y[y == 1]

# Split class 1 into train and test (half-half)
X_train_class_1 = X_class_1[:len(X_class_1) // 2]
y_train_class_1 = y_class_1[:len(y_class_1) // 2]
X_test_class_1 = X_class_1[len(X_class_1) // 2:]
y_test_class_1 = y_class_1[len(y_class_1) // 2:]

# Combine class 0 with the training set and keep the rest for the test set
X_train = np.vstack((X_class_0, X_train_class_1))
y_train = np.hstack((y_class_0, y_train_class_1))
X_test = np.vstack((X_test_class_1))
y_test = np.hstack((y_test_class_1))

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Function to build a neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model with class weights
model = build_model()

# Class weights to handle imbalance
class_weight = {0: 1, 1: 5}  # Give more weight to fraud cases

history = model.fit(X_train_resampled, y_train_resampled, epochs=2, batch_size=32, class_weight=class_weight, validation_split=0.2)

# Function to calculate model performance
def get_model_performance(model, X, y, threshold=0.5):
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > threshold).astype("int32")  # Use threshold tuning
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, precision, recall, f1, y_pred

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Limit to 100 samples for faster SHAP calculations

# Main Streamlit app
st.title("Fraud Detection Model Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Adversarial Attacks", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Performance metrics on clean test data
    clean_acc, clean_precision, clean_recall, clean_f1, y_pred = get_model_performance(model, X_test, y_test, threshold=0.3)  # Adjusted threshold
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot()

    # Visualize fraud vs non-fraud transaction distribution
    st.subheader("Transaction Distribution")
    fraud_count = pd.Series(y_test).value_counts()
    sns.barplot(x=fraud_count.index, y=fraud_count.values)
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    st.pyplot()

# Adversarial Attacks Section (optional)
elif section == "Adversarial Attacks":
    st.header("Adversarial Attacks")
    st.write("This section is optional and can be expanded based on your needs.")

# Explainability Section
elif section == "Explainability":
    st.header("Explainability with SHAP")
    
    # Feature importance plot
    st.subheader("Feature Importance Plot (SHAP)")
    shap_values = explainer.shap_values(X_test[:100])  # Limit X_test for faster visualization
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot()
    
    # Per-transaction explanation
    st.subheader("Per-Transaction Explanation")
    idx = st.slider("Select Transaction Index", 0, len(X_test)-1)
    st.write(f"Transaction: {X_test[idx]}")
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test[idx], matplotlib=True)
    st.pyplot()

# Interactive Prediction Tool Section
elif section == "Interactive Prediction Tool":
    st.header("Interactive Prediction Tool")
    
    # Input features for a new transaction
    st.subheader("Input Transaction Features")
    transaction_input = []
    for i in range(X_test.shape[1]):
        feature_val = st.number_input(f"Feature {i+1}", value=float(X_test[0, i]))
        transaction_input.append(feature_val)
    
    # Predict fraud/not fraud
    transaction_input = np.array(transaction_input).reshape(1, -1)
    pred = (model.predict(transaction_input) > 0.3).astype(int)[0][0]  # Adjusted threshold for prediction
    st.write(f"Prediction: {'Fraud' if pred == 1 else 'Not Fraud'}")
    
    # Show SHAP explanations for the prediction
    st.subheader("Explanation for the Prediction")
    shap_values_input = explainer.shap_values(transaction_input)
    shap.force_plot(explainer.expected_value, shap_values_input[0], transaction_input, matplotlib=True)
    st.pyplot()
