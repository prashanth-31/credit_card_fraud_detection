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
y_class_0 = y[y == 0]
X_class_1 = X[y == 1]
y_class_1 = y[y == 1]

# Split class 1 into train and test (2/3 for training, 1/3 for testing)
n_class_1_train = int(len(X_class_1) * (2 / 3))
n_class_1_test = len(X_class_1) - n_class_1_train

# Split class 0 into train and test (stratify based on the remaining class proportions)
X_train_class_1 = X_class_1[:n_class_1_train]
y_train_class_1 = y_class_1[:n_class_1_train]
X_test_class_1 = X_class_1[n_class_1_train:]
y_test_class_1 = y_class_1[n_class_1_train:]

# For class 0, take a proportionate split of the remaining data
X_train_class_0, X_test_class_0, y_train_class_0, y_test_class_0 = train_test_split(
    X_class_0, y_class_0, test_size=n_class_1_test, random_state=42, stratify=y_class_0)

# Combine the classes back into a single dataset
X_train = np.vstack((X_train_class_0, X_train_class_1))
y_train = np.concatenate((y_train_class_0, y_train_class_1))

X_test = np.vstack((X_test_class_0, X_test_class_1))
y_test = np.concatenate((y_test_class_0, y_test_class_1))

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Function to build a neural network model
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

# Build and train the model
model = build_model()

# Class weights to handle imbalance
class_weight = {0: 1, 1: 5}  # Give more weight to fraud cases

# Adversarial training: Generate adversarial examples and include them in the training set
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)  # Generate Gaussian noise
    X_adv = X + noise  # Add noise to create adversarial examples
    X_adv = np.clip(X_adv, 0, None)  # Ensure no negative values
    return X_adv

X_adv = generate_adversarial_examples(X_train_resampled, epsilon=0.1)

# Combine the original and adversarial examples
X_combined = np.vstack((X_train_resampled, X_adv))
y_combined = np.concatenate((y_train_resampled, y_train_resampled))  # Duplicate the labels

# Train the model with the combined dataset
history = model.fit(X_combined, y_combined, epochs=3, batch_size=32, class_weight=class_weight, validation_split=0.2)

# Function to calculate model performance
def get_model_performance(model, X, y, threshold=0.5):
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > threshold).astype("int32")  # Use threshold tuning
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)  # Handle zero division
    recall = recall_score(y, y_pred, zero_division=0)  # Handle zero division
    f1 = f1_score(y, y_pred, zero_division=0)  # Handle zero division
    return acc, precision, recall, f1, y_pred

# Generate adversarial examples for testing
X_adv_test = generate_adversarial_examples(X_test, epsilon=0.1)
# Normalize adversarial examples to match the training data
X_adv_test = scaler.transform(X_adv_test)

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train_resampled[:100])  # Limit to 100 samples for faster SHAP calculations

# Main Streamlit app
st.title("Fraud Detection Model Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Adversarial Attacks", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Performance metrics on clean test data
    clean_acc, clean_precision, clean_recall, clean_f1, y_pred = get_model_performance(model, X_test, y_test, threshold=0.5)
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")

    # Performance metrics on adversarial test data
    adv_acc, adv_precision, adv_recall, adv_f1, y_pred_adv = get_model_performance(model, X_adv_test, y_test, threshold=0.5)
    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_acc:.4f}")
    st.write(f"Precision: {adv_precision:.4f}")
    st.write(f"Recall: {adv_recall:.4f}")
    st.write(f"F1-Score: {adv_f1:.4f}")

    # Display confusion matrix for clean data
    st.subheader("Confusion Matrix for Clean Data")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Clean Data)")
    st.pyplot()

    # Display confusion matrix for adversarial data
    st.subheader("Confusion Matrix for Adversarial Data")
    cm_adv = confusion_matrix(y_test, y_pred_adv)
    sns.heatmap(cm_adv, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Adversarial Data)")
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
    transaction_input_scaled = scaler.transform(transaction_input)  # Scale the input
    prediction_prob = model.predict(transaction_input_scaled)
    prediction = "Fraud" if prediction_prob[0][0] > 0.5 else "Not Fraud"
    
    st.subheader("Prediction Result")
    st.write(f"Prediction Probability: {prediction_prob[0][0]:.4f}")
    st.write(f"Prediction: {prediction}")
