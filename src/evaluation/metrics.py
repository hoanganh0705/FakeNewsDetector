"""
Evaluation metrics module for fake news detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import os
import json


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: list = None,
    target_names: list = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        labels: List of label values
        target_names: Names for each class
        
    Returns:
        Dictionary containing all metrics
    """
    if target_names is None:
        target_names = ['Real (0)', 'Fake (1)']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
    }
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None).tolist()
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None).tolist()
    
    # ROC-AUC if probabilities are provided
    if y_prob is not None:
        # For binary classification, use probability of positive class
        if len(y_prob.shape) > 1:
            y_prob_positive = y_prob[:, 1]
        else:
            y_prob_positive = y_prob
            
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob_positive)
        metrics['average_precision'] = average_precision_score(y_true, y_prob_positive)
    
    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Results"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"   Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"   F1-Score (macro):  {metrics['f1_macro']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"   ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision:     {metrics['average_precision']:.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"   {'-'*48}")
    
    class_names = ['Real (0)', 'Fake (1)']
    for i, name in enumerate(class_names):
        print(f"   {name:<12} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}")
    
    print(f"\n📉 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"   Actual Real   {cm[0][0]:<7} {cm[0][1]:<7}")
    print(f"   Actual Fake   {cm[1][0]:<7} {cm[1][1]:<7}")
    
    print("=" * 60 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Real (0)', 'Fake (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
):
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
        title: Plot title
    """
    if len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
        title: Plot title
    """
    if len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2, 
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.close()


def save_metrics(metrics: Dict[str, Any], path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {path}")
