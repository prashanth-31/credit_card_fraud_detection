import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


# Load or train your model here
model = tf.keras.models.load_model('fraud_detection_model.h5')


data = pd.read_csv('creditcard.csv')
st.write(data)

# Load the features and labels
X_test = data.drop('Class', axis=1).values
y_test = data['Class'].values

# Generate adversarial examples (as before)
def generate_adversarial_examples(model, X_test, y_test, epsilon=0.1):
    # Convert inputs to a tensor
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Create a gradient tape to record operations
    with tf.GradientTape() as tape:
        # Make predictions on the test set
        tape.watch(X_test_tensor)
        predictions = model(X_test_tensor)
        loss = tf.keras.losses.binary_crossentropy(y_test_tensor, predictions)
    
    # Calculate gradients of the loss with respect to the input
    gradients = tape.gradient(loss, X_test_tensor)

    # Generate adversarial examples by adding perturbations to the original input
    X_adv = X_test + epsilon * tf.sign(gradients)

    # Clip the values to ensure they remain within the valid range
    X_adv = tf.clip_by_value(X_adv, 0, 1)  # Assuming inputs are normalized to [0, 1]

    return X_adv.numpy(), y_test  # Return as numpy arrays


# Function to calculate model performance
def get_model_performance(model, X, y):
    y_pred = model.predict(X)

    # Reshape the predictions array (from (75000, 1) to (75000,))
    y_pred = y_pred.ravel()  # Use .ravel() to flatten the array

    # For binary classification, convert probabilities to 0 or 1 predictions
    y_pred = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return acc, precision, recall, f1


# Example usage: generate adversarial examples
X_adv, y_adv = generate_adversarial_examples(model, X_test, y_test)

# Main app
st.title("Fraud Detection Model Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Adversarial Attacks", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Display performance metrics on clean data
    clean_acc, clean_precision, clean_recall, clean_f1 = get_model_performance(model, X_test, y_test)
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")
    
    # Display performance metrics on adversarial data
    adv_acc, adv_precision, adv_recall, adv_f1 = get_model_performance(model, X_adv, y_adv)
    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_acc:.4f}")
    st.write(f"Precision: {adv_precision:.4f}")
    st.write(f"Recall: {adv_recall:.4f}")
    st.write(f"F1-Score: {adv_f1:.4f}")

    # Visualize fraud vs non-fraud transaction distribution
    st.subheader("Transaction Distribution")
    fraud_count = pd.Series(y_test).value_counts()
    sns.barplot(x=fraud_count.index, y=fraud_count.values)
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    st.pyplot()

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
    original_pred = model.predict([X_test[idx]])[0]
    adv_pred = model.predict([X_adv[idx]])[0]
    st.write(f"Original Prediction: {'Fraud' if original_pred > 0.5 else 'Not Fraud'}")
    st.write(f"Adversarial Prediction: {'Fraud' if adv_pred > 0.5 else 'Not Fraud'}")

# Explainability Section
elif section == "Explainability":
    st.header("Explainability with SHAP")
    
    # Feature importance plot
    st.subheader("Feature Importance Plot (SHAP)")
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot()
    
    # Per-transaction explanation
    st.subheader("Per-Transaction Explanation")
    idx = st.slider("Select Transaction Index", 0, len(X_test)-1)
    st.write(f"Transaction: {X_test[idx]}")
    shap.force_plot(explainer.expected_value[1], shap_values[1][idx], X_test[idx], matplotlib=True)
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
    pred = model.predict(transaction_input)[0]
    st.write(f"Prediction: {'Fraud' if pred > 0.5 else 'Not Fraud'}")
    
    # Show SHAP explanations for the prediction
    shap_values_input = explainer.shap_values(transaction_input)
    shap.force_plot(explainer.expected_value[1], shap_values_input[1], transaction_input, matplotlib=True)
    st.pyplot()
