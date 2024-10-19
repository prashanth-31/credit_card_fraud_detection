import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Load dataset
data = pd.read_csv('creditcard.csv')  # Adjust with your dataset path
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target (fraud/not fraud)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training set
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
class_weight = {0: 1, 1: 5}  # Give more weight to fraud cases

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=3, batch_size=32, class_weight=class_weight, validation_split=0.2)

# Function to calculate model performance
def get_model_performance(model, X, y):
    y_pred = (model.predict(X) > 0.5).astype("int32")  # Assuming binary classification with sigmoid
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, precision, recall, f1

# Calculate initial performance metrics on clean data
clean_acc, clean_precision, clean_recall, clean_f1 = get_model_performance(model, X_test, y_test)

# Function to create adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)  # Generate Gaussian noise
    X_adv = X + noise  # Add noise to create adversarial examples
    X_adv = np.clip(X_adv, 0, None)  # Ensure no negative values
    return X_adv

# Generate adversarial examples
X_adv = generate_adversarial_examples(X_test)
y_adv = y_test  # Assuming labels remain the same for this example

# Calculate performance metrics on adversarial data
adv_acc, adv_precision, adv_recall, adv_f1 = get_model_performance(model, X_adv, y_adv)

# Main Streamlit app
st.title("Fraud Detection Model Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Explainability", "Interactive Prediction Tool", "Adversarial Attacks"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")

    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_acc:.4f}")
    st.write(f"Precision: {adv_precision:.4f}")
    st.write(f"Recall: {adv_recall:.4f}")
    st.write(f"F1-Score: {adv_f1:.4f}")

    # Visualize confusion matrix for clean data
    st.subheader("Confusion Matrix for Clean Data")
    cm = confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Clean Data)")
    st.pyplot()

    # Visualize confusion matrix for adversarial data
    st.subheader("Confusion Matrix for Adversarial Data")
    cm_adv = confusion_matrix(y_adv, (model.predict(X_adv) > 0.5).astype(int))
    sns.heatmap(cm_adv, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Adversarial Data)")
    st.pyplot()

# Adversarial Attacks Section
elif section == "Adversarial Attacks":
    st.header("Adversarial Attacks")
    
    # Before vs. After Attack Comparison
    st.subheader("Before vs. After Attack")
    st.write("Model accuracy before attack: ", round(clean_acc,2))
    st.write("Model accuracy after attack: ", round(adv_acc,2))
    
    # Generate adversarial example
    st.subheader("Adversarial Example")
    idx = st.slider("Select Transaction Index", 0, len(X_adv)-1)
    st.write(f"Original Transaction: {X_test[idx]}")
    st.write(f"Adversarial Transaction: {X_adv[idx]}")
    
    # Reshape input for prediction
    original_input = X_test[idx:idx+1]  # Ensure correct shape for model input
    adversarial_input = X_adv[idx:idx+1]  # Ensure correct shape for model input
    
    # Get predictions
    original_pred = (model.predict(original_input) > 0.5).astype(int)[0][0]  # Reshape input
    adv_pred = (model.predict(adversarial_input) > 0.5).astype(int)[0][0]  # Reshape input
    
    # Indicate if the original prediction is fraud
    if original_pred == 1:
        st.success("The original transaction is classified as Fraud.")
    else:
        st.warning("The original transaction is classified as Not Fraud.")

    # Indicate if the adversarial prediction is fraud
    if adv_pred == 1:
        st.error("The adversarial transaction is classified as Fraud.")
    else:
        st.info("The adversarial transaction is classified as Not Fraud.")


# Explainability Section
elif section == "Explainability":
    st.header("Explainability with Seaborn")
    
    # Calculate feature importances using basic correlations
    st.subheader("Feature Importance Plot (Correlation with Target)")
    feature_importance = pd.DataFrame({
        'Feature': data.columns[:-1],
        'Importance': np.abs(np.corrcoef(X_train_resampled.T, y_train_resampled)[-1, :-1])  # Absolute correlation between features and target
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title("Feature Importance Based on Correlation with Target")
    st.pyplot()

    # Per-Transaction Explanation: Show feature values for a selected transaction
    st.subheader("Per-Transaction Feature Values")
    idx = st.slider("Select Transaction Index", 0, len(X_test) - 1)
    selected_transaction = pd.DataFrame(X_test[idx], index=data.columns[:-1], columns=["Feature Value"])
    
    st.write(f"Transaction {idx}: Feature Values")
    st.dataframe(selected_transaction.T)


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
    pred_prob = model.predict(transaction_input_scaled)[0][0]
    pred_label = "Fraud" if pred_prob > 0.5 else "Not Fraud"
    
    st.write(f"Prediction: {pred_label}")
