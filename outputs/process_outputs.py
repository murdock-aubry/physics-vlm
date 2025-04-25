import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, log_loss
import numpy as np

def yes_no_to_bool(response: str) -> bool:
    """Converts 'yes'/'no' string to boolean. Defaults to False for anything else."""
    return str(response).strip().lower() == "yes"

def process_binary_outputs(df):
    """Process outputs for binary classification."""
    # Clean and convert columns if needed
    df['label'] = df['label'].astype(bool)
    df['response'] = df['response'].str.strip().str.lower().apply(yes_no_to_bool)

    # Extract prediction and target values
    y_true = df['label']
    y_pred = df['response']

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # Display results
    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"🔁 Recall: {recall:.3f}")
    print("\n📊 Confusion Matrix:")
    print(conf_matrix)
    print("\n🧾 Full Classification Report:")
    print(report)

def process_continuous_outputs(df):
    """Process outputs for continuous probability classification."""
    # Ensure yes_prob and no_prob add to 1
    total_prob = df['yes_prob'] + df['no_prob']
    df['yes_prob'] = df['yes_prob'] / total_prob
    df['no_prob'] = df['no_prob'] / total_prob

    # Extract true labels and predicted probabilities
    y_true = df['label'].astype(int)  # Convert label to 1/0
    y_pred = df['yes_prob']  # Use yes_prob for log-loss (since no_prob is complementary)

    # Compute Log-Loss (Cross-Entropy Loss)
    log_loss_value = log_loss(y_true, y_pred)
    
    # Display result
    print(f"Log-Loss: {log_loss_value:.3f}")

def process_outputs(mode='continuous', file_path="resp_intphys_O2_nat-2000-continous.csv"):
    """Process outputs based on the selected mode ('binary' or 'continuous')."""
    # Load CSV
    df = pd.read_csv(file_path)

    if mode == 'binary':
        print("Processing as Binary Outputs:")
        process_binary_outputs(df)
    elif mode == 'continuous':
        print("Processing as Continuous Outputs:")
        process_continuous_outputs(df)
    else:
        print("Invalid mode. Please choose either 'binary' or 'continuous'.")
        
if __name__ == "__main__":
    process_outputs()