import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Optional, Union, Callable

class WildfirePredictionLoss(tf.keras.losses.Loss):
    """
    Custom loss function combining multiple components for wildfire prediction:
    - Binary cross-entropy for fire/no-fire classification
    - Spatial continuity loss to enforce smooth predictions
    - Temporal consistency loss to ensure temporal coherence
    - Edge detection loss to preserve fire boundaries
    """
    def __init__(
        self,
        spatial_weight: float = 0.4,
        temporal_weight: float = 0.3,
        edge_weight: float = 0.3,
        focal_gamma: float = 2.0,
        class_weights: Optional[dict] = None
    ):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.edge_weight = edge_weight
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Focal loss for handling class imbalance
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal weights
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha = tf.where(tf.equal(y_true, 1),
                        self.class_weights[1],
                        self.class_weights[0])
        focal_weight = alpha * tf.pow(1 - p_t, self.focal_gamma)
        
        # Calculate binary cross-entropy
        bce = -tf.math.log(p_t)
        
        return tf.reduce_mean(focal_weight * bce)

    def spatial_continuity_loss(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss term to enforce spatial smoothness in predictions
        """
        # Calculate gradients in both spatial dimensions
        grad_y = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        grad_x = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        
        # Calculate total variation
        return (tf.reduce_mean(tf.abs(grad_y)) + 
                tf.reduce_mean(tf.abs(grad_x)))

    def temporal_consistency_loss(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss term to enforce temporal consistency between consecutive frames
        """
        if len(y_pred.shape) < 4:
            return 0.0
            
        # Calculate temporal differences
        temp_diff = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        return tf.reduce_mean(tf.abs(temp_diff))

    def edge_detection_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss term to preserve fire boundaries using Sobel edge detection
        """
        # Sobel filters for edge detection
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
        
        # Reshape filters for conv2d operation
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Detect edges in both prediction and ground truth
        pred_edges_x = tf.nn.conv2d(y_pred[..., tf.newaxis], sobel_x, 
                                  strides=[1, 1, 1, 1], padding='SAME')
        pred_edges_y = tf.nn.conv2d(y_pred[..., tf.newaxis], sobel_y, 
                                  strides=[1, 1, 1, 1], padding='SAME')
        true_edges_x = tf.nn.conv2d(y_true[..., tf.newaxis], sobel_x, 
                                  strides=[1, 1, 1, 1], padding='SAME')
        true_edges_y = tf.nn.conv2d(y_true[..., tf.newaxis], sobel_y, 
                                  strides=[1, 1, 1, 1], padding='SAME')
        
        # Calculate edge intensity
        pred_edges = tf.sqrt(tf.square(pred_edges_x) + tf.square(pred_edges_y))
        true_edges = tf.sqrt(tf.square(true_edges_x) + tf.square(true_edges_y))
        
        return tf.reduce_mean(tf.abs(pred_edges - true_edges))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Combine all loss components
        """
        # Ensure tensors have proper shape
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate individual loss components
        focal = self.focal_loss(y_true, y_pred)
        spatial = self.spatial_continuity_loss(y_pred)
        temporal = self.temporal_consistency_loss(y_pred)
        edge = self.edge_detection_loss(y_true, y_pred)
        
        # Combine losses with weights
        total_loss = (focal +
                     self.spatial_weight * spatial +
                     self.temporal_weight * temporal +
                     self.edge_weight * edge)
        
        return total_loss

class UncertaintyLoss(tf.keras.losses.Loss):
    """
    Loss function for uncertainty estimation in predictions
    """
    def __init__(self, reduction: str = tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)

    def call(self, y_true: tf.Tensor, y_pred_with_uncertainty: tuple) -> tf.Tensor:
        """
        Calculate loss incorporating prediction uncertainty
        
        Args:
            y_true: Ground truth values
            y_pred_with_uncertainty: Tuple of (predictions, uncertainty)
        """
        y_pred, uncertainty = y_pred_with_uncertainty
        
        # Calculate negative log likelihood with uncertainty
        nll = 0.5 * tf.math.log(uncertainty) + \
              0.5 * tf.math.divide(tf.square(y_true - y_pred), uncertainty)
        
        return tf.reduce_mean(nll)

def get_loss_function(
    loss_type: str = 'wildfire',
    **kwargs
) -> Union[WildfirePredictionLoss, UncertaintyLoss, Callable]:
    """
    Factory function to get the appropriate loss function
    
    Args:
        loss_type: Type of loss function ('wildfire', 'uncertainty', or 'combined')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'wildfire':
        return WildfirePredictionLoss(**kwargs)
    elif loss_type == 'uncertainty':
        return UncertaintyLoss(**kwargs)
    elif loss_type == 'combined':
        # Create a combined loss function
        wildfire_loss = WildfirePredictionLoss(**kwargs)
        uncertainty_loss = UncertaintyLoss()
        
        def combined_loss(y_true, y_pred):
            return wildfire_loss(y_true, y_pred[0]) + uncertainty_loss(y_true, y_pred)
            
        return combined_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")