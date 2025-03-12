import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_results(history=None, y_true=None, y_pred=None, model_name="Model", output_dir="outputs"):
    """
    Parameters:
        history (dict): Training history containing 'loss' and optionally 'val_loss'.
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the model (for title purposes).
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    
    
    # Plot confusion matrix
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[1], cmap='Blues')
        axes[1].set_title(f'{model_name} Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    else:
        axes[1].set_visible(False)
    
