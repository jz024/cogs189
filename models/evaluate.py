# models/evaluate.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, X, y, label=""):
    """Evaluate model performance and print metrics."""
    # Get model predictions (probabilities)
    probs = model.predict(X)
    
    # Convert probabilities to binary predictions using 0.5 threshold
    preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y, preds)
    print(f"{label} Accuracy: {acc:.4f}")
    
    print(f"Confusion Matrix ({label}):")
    print(confusion_matrix(y, preds))
    
    return acc, preds
