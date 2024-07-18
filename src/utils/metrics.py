import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    cohen_kappa_score
)
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class WildfireMetrics:
    """
    Comprehensive metrics for evaluating wildfire prediction performance.
    Includes spatial, temporal, and uncertainty metrics.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            save_dir: Directory to save metric plots and results
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate all available metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            uncertainties: Prediction uncertainties (optional)
            threshold: Classification threshold
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self.classification_metrics(y_true, y_pred, threshold))
        
        # Calibration metrics
        metrics.update(self.calibration_metrics(y_true, y_pred))
        
        # Spatial metrics
        metrics.update(self.spatial_metrics(y_true, y_pred))
        
        # Temporal metrics if data has temporal dimension
        if len(y_true.shape) > 2:
            metrics.update(self.temporal_metrics(y_true, y_pred))
        
        # Uncertainty metrics if uncertainties provided
        if uncertainties is not None:
            metrics.update(self.uncertainty_metrics(y_true, y_pred, uncertainties))
        
        return metrics

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate basic classification metrics.
        """
        # Convert predictions to binary
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred_binary.flatten()).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true.flatten(), y_pred_binary.flatten())
        kappa = cohen_kappa_score(y_true.flatten(), y_pred_binary.flatten())
        
        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true.flatten(),
            y_pred.flatten()
        )
        pr_auc = auc(recall_curve, precision_curve)
        
        if self.save_dir:
            self._plot_roc_curve(fpr, tpr, roc_auc)
            self._plot_pr_curve(recall_curve, precision_curve, pr_auc)
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'kappa': kappa,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

    def calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Calculate prediction calibration metrics.
        """
        # Create bins for calibration
        bins = np.linspace(0, 1, n_bins + 1)
        binned = np.digitize(y_pred.flatten(), bins) - 1
        
        # Calculate calibration metrics
        bin_accs = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)
        bin_sizes = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = binned == i
            if np.any(mask):
                bin_accs[i] = np.mean(y_true.flatten()[mask])
                bin_confs[i] = np.mean(y_pred.flatten()[mask])
                bin_sizes[i] = np.sum(mask)
        
        # Calculate ECE (Expected Calibration Error)
        ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_sizes / len(binned)))
        
        if self.save_dir:
            self._plot_calibration_curve(bin_accs, bin_confs, bin_sizes)
        
        return {
            'ece': ece,
            'bin_accuracies': bin_accs.tolist(),
            'bin_confidences': bin_confs.tolist(),
            'bin_sizes': bin_sizes.tolist()
        }

    def spatial_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate metrics for spatial accuracy.
        """
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate IoU (Intersection over Union)
        intersection = np.sum(y_true * y_pred_binary)
        union = np.sum((y_true + y_pred_binary) > 0)
        iou = intersection / union if union > 0 else 0
        
        # Calculate centroid error
        true_centroids = self._calculate_centroids(y_true)
        pred_centroids = self._calculate_centroids(y_pred_binary)
        centroid_error = np.mean([
            np.linalg.norm(tc - pc)
            for tc, pc in zip(true_centroids, pred_centroids)
            if tc is not None and pc is not None
        ])
        
        # Calculate boundary F1 score
        boundary_f1 = self._calculate_boundary_f1(y_true, y_pred_binary)
        
        return {
            'iou': iou,
            'centroid_error': centroid_error,
            'boundary_f1': boundary_f1
        }

    def temporal_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate metrics for temporal consistency.
        """
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate temporal consistency
        temp_diff_true = np.abs(y_true[1:] - y_true[:-1])
        temp_diff_pred = np.abs(y_pred_binary[1:] - y_pred_binary[:-1])
        temporal_consistency = np.mean(temp_diff_true == temp_diff_pred)
        
        # Calculate prediction delay
        delay = self._calculate_prediction_delay(y_true, y_pred_binary)
        
        return {
            'temporal_consistency': temporal_consistency,
            'prediction_delay': delay
        }

    def uncertainty_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict:
        """
        Calculate metrics related to prediction uncertainty.
        """
        # Calculate error-uncertainty correlation
        errors = np.abs(y_true - y_pred)
        error_uncertainty_corr = np.corrcoef(errors.flatten(), uncertainties.flatten())[0, 1]
        
        # Calculate AUCE (Area Under Confidence-Error curve)
        auce = self._calculate_auce(errors, uncertainties)
        
        # Calculate calibration of uncertainty estimates
        uncertainty_calibration = self._calculate_uncertainty_calibration(
            y_true,
            y_pred,
            uncertainties
        )
        
        return {
            'error_uncertainty_corr': error_uncertainty_corr,
            'auce': auce,
            'uncertainty_calibration': uncertainty_calibration
        }

    def _calculate_centroids(self, binary_masks: np.ndarray) -> List[Optional[np.ndarray]]:
        """Calculate centroids for each binary mask."""
        centroids = []
        for mask in binary_masks:
            if np.any(mask):
                coords = np.where(mask)
                centroids.append(np.array([np.mean(coords[0]), np.mean(coords[1])]))
            else:
                centroids.append(None)
        return centroids

    def _calculate_boundary_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        boundary_width: int = 2
    ) -> float:
        """Calculate F1 score for boundary predictions."""
        from scipy.ndimage import binary_dilation
        
        # Extract boundaries
        true_boundaries = binary_dilation(y_true, iterations=boundary_width) ^ y_true
        pred_boundaries = binary_dilation(y_pred, iterations=boundary_width) ^ y_pred
        
        return f1_score(
            true_boundaries.flatten(),
            pred_boundaries.flatten(),
            zero_division=1
        )

    def _calculate_prediction_delay(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate average prediction delay in time steps."""
        delays = []
        for i in range(len(y_true)):
            if np.any(y_true[i]):
                true_start = np.where(y_true[i])[0][0]
                if np.any(y_pred[i]):
                    pred_start = np.where(y_pred[i])[0][0]
                    delays.append(pred_start - true_start)
        
        return np.mean(delays) if delays else 0

    def _calculate_auce(
        self,
        errors: np.ndarray,
        uncertainties: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Area Under Confidence-Error curve."""
        # Sort by uncertainty
        sorted_idx = np.argsort(uncertainties.flatten())
        sorted_errors = errors.flatten()[sorted_idx]
        
        # Calculate cumulative errors
        cum_errors = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
        
        # Calculate AUC
        return auc(np.linspace(0, 1, len(cum_errors)), cum_errors)

    def _calculate_uncertainty_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate calibration score for uncertainty estimates."""
        errors = np.abs(y_true - y_pred)
        
        # Create confidence bins
        confidences = 1 - uncertainties
        bins = np.linspace(0, 1, n_bins + 1)
        binned = np.digitize(confidences.flatten(), bins) - 1
        
        # Calculate calibration
        cal_errors = np.zeros(n_bins)
        for i in range(n_bins):
            mask = binned == i
            if np.any(mask):
                cal_errors[i] = np.abs(
                    np.mean(errors.flatten()[mask]) - 
                    (1 - np.mean(confidences.flatten()[mask]))
                )
        
        return np.mean(cal_errors)

    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
        """Plot ROC curve."""
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'roc_curve.png')
        plt.close()

    def _plot_pr_curve(
        self,
        recall: np.ndarray,
        precision: np.ndarray,
        pr_auc: float
    ):
        """Plot Precision-Recall curve."""
        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'pr_curve.png')
        plt.close()

    def _plot_calibration_curve(
        self,
        accuracies: np.ndarray,
        confidences: np.ndarray,
        sizes: np.ndarray
    ):
        """Plot calibration curve."""
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(confidences, accuracies, 'ro-', label='Model')
        
        # Plot histogram of predictions
        plt.hist(confidences, weights=sizes/np.sum(sizes), bins=10, 
                alpha=0.3, label='Distribution')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'calibration_curve.png')
        plt.close()