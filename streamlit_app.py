import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('fraud_detection_model.h5')

# Load dataset
data = pd.read_csv('creditcard.csv')  # Replace with your actual dataset path
X = data.iloc[:, :-1].values  # Adjust to your actual features
y = data.iloc[:, -1].values  # Adjust to your actual target column

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Function to create adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)  # Generate Gaussian noise
    X_adv = X + noise  # Add noise to create adversarial examples
    X_adv = np.clip(X_adv, 0, None)  # Ensure no negative values
    return X_adv

# Generate adversarial examples
X_adv = generate_adversarial_examples(X_test)
y_adv = y_test  # Assuming labels remain the same for this example

# Function to calculate model performance
def get_model_performance(model, X, y, threshold=0.2):  # Adjust threshold lower than 0.5
    y_pred = (model.predict(X) > threshold).astype("int32")  # Adjust threshold
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    return acc, precision, recall, f1, y_pred

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    st.pyplot()

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train_smote[:100])  # Limit X_train for faster SHAP calculation

# Main app
st.title("Fraud Detection Model Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Adversarial Attacks", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")

    # Adjust threshold using a slider
    st.subheader("Adjust Prediction Threshold")
    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.2, 0.01)  # Default lower threshold at 0.2

    # Display performance metrics on clean data
    clean_acc, clean_precision, clean_recall, clean_f1, y_pred_clean = get_model_performance(model, X_test, y_test, threshold)
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")

    # Display performance metrics on adversarial data
    adv_acc, adv_precision, adv_recall, adv_f1, y_pred_adv = get_model_performance(model, X_adv, y_adv, threshold)
    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_acc:.4f}")
    st.write(f"Precision: {adv_precision:.4f}")
    st.write(f"Recall: {adv_recall:.4f}")
    st.write(f"F1-Score: {adv_f1:.4f}")

    # Visualize fraud vs non-fraud transaction distribution
    st.subheader("Class Distribution in Test Set")
    class_distribution = pd.Series(y_test).value_counts()
    st.write(class_distribution)

    # Confusion Matrix for clean data
    st.subheader("Confusion Matrix on Clean Data")
    plot_confusion_matrix(y_test, y_pred_clean)

    # ROC Curve
    st.subheader("ROC Curve on Clean Data")
    plot_roc_curve(y_test, model.predict(X_test))

# Adversarial Attacks Section
elif section == "Adversarial Attacks":
    st.header("Adversarial Attacks")

    # Before vs. After Attack Comparison
    st.subheader("Before vs. After Attack")
    st.write("Model accuracy before attack: ", clean_acc)
    st.write("Model accuracy after attack: ", adv_acc)

    # Generate adversarial example
    st.subheader("Adversarial Example")
    idx = st.slider("Select Transaction Index", 0, len(X_adv)-1)
    st.write(f"Original Transaction: {X_test[idx]}")
    st.write(f"Adversarial Transaction: {X_adv[idx]}")
    original_pred = (model.predict(np.array([X_test[idx]])) > threshold).astype(int)[0][0]
    adv_pred = (model.predict(np.array([X_adv[idx]])) > threshold).astype(int)[0][0]
    st.write(f"Original Prediction: {'Fraud' if original_pred == 1 else 'Not Fraud'}")
    st.write(f"Adversarial Prediction: {'Fraud' if adv_pred == 1 else 'Not Fraud'}")

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

    # Input features for new transaction
    st.subheader("Input Transaction Features")
    transaction_input = []
    for i in range(X_test.shape[1]):
        feature_val = st.number_input(f"Feature {i+1}", value=float(X_test[0, i]))
        transaction_input.append(feature_val)

    # Predict fraud/not fraud
    transaction_input = np.array(transaction_input).reshape(1, -1)
    pred = (model.predict(transaction_input) > threshold).astype(int)[0][0]
    st.write(f"Prediction: {'Fraud' if pred == 1 else 'Not Fraud'}")

    # Show SHAP explanations for the prediction
    st.subheader("Explanation for the Prediction")
    shap_values_input = explainer.shap_values(transaction_input)
    shap.force_plot(explainer.expected_value, shap_values_input[0], transaction_input, matplotlib=True)
    st.pyplot()
