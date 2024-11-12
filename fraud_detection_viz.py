"""
Fraud Detection Visualization Module
----------------------------------
This module provides comprehensive visualization capabilities for fraud detection analysis,
including confusion matrices, ROC curves, precision-recall curves, and feature importance plots.
It uses matplotlib and seaborn with custom styling for professional-grade visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class FraudVisualization:
    """
    A class containing methods for visualizing fraud detection results.
    """
    
    def __init__(self):
        """Initialize visualization settings with professional plotting style."""
        self.set_style()
    
    @staticmethod
    def set_style():
        """Set global matplotlib parameters for consistent, professional-looking plots."""
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'figure.dpi': 300,
            'figure.figsize': (10, 6),
            'font.size': 12,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.title_fontsize': 12
        })

    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot a styled confusion matrix heatmap.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create custom colormap
        cmap = plt.cm.Blues
        norm = plt.Normalize(vmin=0, vmax=np.max(cm))
        
        # Create the heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=',d',
            cmap=cmap,
            norm=norm,
            square=True,
            cbar=True,
            cbar_kws={'label': 'Count'},
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud']
        )
        
        plt.title('Confusion Matrix', pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot the ROC curve with AUC score.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred_proba (numpy.ndarray): Predicted probabilities
            save_path (str, optional): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, 
            tpr, 
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot the Precision-Recall curve.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred_proba (numpy.ndarray): Predicted probabilities
            save_path (str, optional): Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color='purple',
            lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})'
        )
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_feature_importance(self, model, feature_names, save_path=None):
        """
        Plot feature importance for each base model in the stacking classifier.
        
        Args:
            model: Trained stacking classifier model
            feature_names (list): List of feature names
            save_path (str, optional): Path to save the plot
        """
        for name, clf in model.named_estimators_.items():
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance - {name}')
                
                # Create bar plot
                bars = plt.bar(
                    range(len(importances)),
                    importances[indices],
                    align="center",
                    color='lightblue',
                    edgecolor='navy'
                )
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom'
                    )
                
                plt.xticks(
                    range(len(importances)),
                    [feature_names[i] for i in indices],
                    rotation=45,
                    ha='right'
                )
                plt.xlabel('Features')
                plt.ylabel('Importance Score')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(f'{save_path}_{name}.png', bbox_inches='tight', dpi=300)
                plt.show()
            else:
                print(f"Model '{name}' does not support feature importances.")

    def plot_metrics_dashboard(self, metrics, save_path=None):
        """
        Create a comprehensive dashboard of all key metrics.
        
        Args:
            metrics (dict): Dictionary containing all evaluation metrics
            save_path (str, optional): Path to save the plot
        """
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Fraud Detection Performance Dashboard', fontsize=16)
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt=',d',
            cmap='Blues',
            square=True,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_pred_proba'])
        ax2.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_title('ROC Curve')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        
        # Precision-Recall Curve
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(
            metrics['y_true'],
            metrics['y_pred_proba']
        )
        ax3.plot(recall, precision, 'g-')
        ax3.set_title('Precision-Recall Curve')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        
        # Key Metrics Text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        metrics_text = (
            f'Key Metrics:\n\n'
            f'Sensitivity (TPR): {metrics["sensitivity"]:.3f}\n'
            f'Specificity (TNR): {metrics["specificity"]:.3f}\n'
            f'ROC AUC: {metrics["roc_auc"]:.3f}\n'
            f'Precision: {metrics["precision"]:.3f}\n'
            f'F1 Score: {metrics["f1_score"]:.3f}'
        )
        ax4.text(
            0.1,
            0.7,
            metrics_text,
            fontsize=12,
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

def create_visualization_report(fraud_detector, metrics, output_dir=None):
    """
    Create a comprehensive visualization report for a fraud detection model.
    
    Args:
        fraud_detector: Trained FraudDetectionSystem instance
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str, optional): Directory to save visualization files
    """
    viz = FraudVisualization()
    
    # Create visualizations
    viz.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=f"{output_dir}/confusion_matrix.png" if output_dir else None
    )
    
    viz.plot_roc_curve(
        metrics['y_true'],
        metrics['y_pred_proba'],
        save_path=f"{output_dir}/roc_curve.png" if output_dir else None
    )
    
    viz.plot_precision_recall_curve(
        metrics['y_true'],
        metrics['y_pred_proba'],
        save_path=f"{output_dir}/pr_curve.png" if output_dir else None
    )
    
    viz.plot_feature_importance(
        fraud_detector.model,
        fraud_detector.features,
        save_path=f"{output_dir}/feature_importance" if output_dir else None
    )
    
    viz.plot_metrics_dashboard(
        metrics,
        save_path=f"{output_dir}/dashboard.png" if output_dir else None
    )

