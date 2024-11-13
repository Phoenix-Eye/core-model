import tensorflow as tf
from tensorflow import keras
from keras import (
    layers, 
    Model, 
    Input, 
    backend as K, 
    mixed_precision
)
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
import tensorflow_probability as tfp
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score
)
from typing import List, Tuple, Dict, Optional, Any
import time
import json
from pathlib import Path
from .layers import (
    SpatialAttentionLayer,
    TemporalAttentionLayer,
    SpatioTemporalAttention,
    UncertaintyLayer,
    ResidualConvLSTM2D,
    FeatureFusion
)
from .losses import WildfirePredictionLoss, UncertaintyLoss

from .layers import (
    SpatialAttentionLayer,
    TemporalAttentionLayer,
    SpatioTemporalAttention,
    UncertaintyLayer,
    ResidualConvLSTM2D,
    FeatureFusion
)
from .losses import WildfirePredictionLoss, UncertaintyLoss

class WildfirePredictionModel:
    """
    Enhanced wildfire prediction model with uncertainty estimation and attention mechanisms.
    
    Features:
    - Dual-stream architecture (spatial + temporal)
    - Transfer learning with ResNet50V2
    - Attention mechanisms for both spatial and temporal data
    - Uncertainty estimation
    - Ensemble capabilities
    """
    
    def __init__(
        self,
        config: Dict,
        num_ensemble: int = 3,
        uncertainty: bool = True,
        transfer_learning: bool = True
    ):
        # GPU and memory optimization setup
        if tf.config.list_physical_devices('GPU'):
            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # Memory growth
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable XLA optimization
        tf.config.optimizer.set_jit(True)
        
        # Initialize as before
        if isinstance(config, dict) and 'model' in config:
            self.config = config['model']
        else:
            self.config = config
        self.num_ensemble = num_ensemble
        self.uncertainty = uncertainty
        self.transfer_learning = transfer_learning
        self.models = []
        
    def build_feature_extractor(self) -> Model:
        """
        Builds the spatial feature extraction branch using transfer learning.
        """
        if self.transfer_learning:
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape'][1:],
                dtype='float16'  # Use mixed precision
            )
            # Freeze early layers
            for layer in base_model.layers[:100]:
                layer.trainable = False
                
            # Add spatial reduction layers for memory efficiency
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(512, activation='relu', dtype='float16')(x)
            
            return Model(base_model.input, x)
        else:
            # Custom feature extraction
            inputs = Input(shape=self.config['input_shape'][1:])
            
            # Initial reduction
            x = layers.Conv2D(64, (7, 7), strides=2, padding='same', dtype='float16')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
            
            # Residual blocks with progressive downsampling
            for filters in [64, 128, 256, 512]:
                x = self._residual_block(x, filters, downsample=True)
            
            # Global pooling
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(512, activation='relu', dtype='float16')(x)
            
            return Model(inputs, x)
        
    def _residual_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """
        Creates a residual block for the feature extractor.
        """
        shortcut = x
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        
        return x
        
    def build_single_model(self) -> Model:
        """
        Builds a single model instance with spatial and temporal branches.
        """
        # Spatial input branch
        spatial_input = Input(shape=self.config['input_shape'], name='spatial_input')
        
        # Apply TimeDistributed wrapper to feature extractor
        feature_extractor = self.build_feature_extractor()
        spatial_features = layers.TimeDistributed(feature_extractor)(spatial_input)
        
        # Add spatiotemporal attention
        st_attention = SpatioTemporalAttention()
        spatial_features, _ = st_attention(spatial_features)
        
        # ConvLSTM layers for spatial-temporal processing
        x = ResidualConvLSTM2D(64, (5, 5))(spatial_features)
        x = layers.BatchNormalization()(x)
        
        x = ResidualConvLSTM2D(64, (3, 3))(x)
        x = layers.BatchNormalization()(x)
        
        x = ResidualConvLSTM2D(64, (1, 1))(x)
        spatial_output = layers.BatchNormalization()(x)
        
        # Temporal input branch
        temporal_input = Input(
            shape=(None, self.config['num_features']),
            name='temporal_input'
        )
        
        # LSTM layers with temporal attention
        temp_attention = TemporalAttentionLayer()
        y, _ = temp_attention(temporal_input)
        
        y = layers.LSTM(128, return_sequences=True)(y)
        y = layers.Dropout(0.2)(y)
        
        y = layers.LSTM(64, return_sequences=True)(y)
        temporal_output = layers.Dropout(0.2)(y)
        
        # Feature fusion
        fusion = FeatureFusion(128)
        combined = fusion([
            layers.Flatten()(spatial_output),
            temporal_output
        ])
        
        # Final prediction layers
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.Dropout(0.2)(z)
        
        if self.uncertainty:
            # Output both prediction and uncertainty
            uncertainty_layer = UncertaintyLayer()
            predictions, uncertainty = uncertainty_layer(z)
            outputs = [predictions, uncertainty]
        else:
            # Output only predictions
            predictions = layers.Dense(1, activation='sigmoid')(z)
            outputs = predictions
            
        # Create model
        model = Model(
            inputs=[spatial_input, temporal_input],
            outputs=outputs
        )
        
        # Compile model
        loss = (UncertaintyLoss() if self.uncertainty 
                else WildfirePredictionLoss())
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
        
    def build_ensemble(self):
        """
        Builds an ensemble of models.
        """
        self.models = [
            self.build_single_model()
            for _ in range(self.num_ensemble)
        ]
        
    def fit(
        self,
        x: List[np.ndarray],
        y: np.ndarray,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Trains all models in the ensemble with gradient accumulation.
        """
        # Calculate accumulation steps based on virtual batch size
        virtual_batch_size = 256 
        actual_batch_size = kwargs.get('batch_size', 32)
        accumulation_steps = virtual_batch_size // actual_batch_size
        
        strategies = tf.distribute.MirroredStrategy()
        with strategies.scope():
            histories = []
            for i, model in enumerate(self.models):
                print(f"\nTraining model {i+1}/{self.num_ensemble}")
                
                history = self.train_with_gradient_accumulation(
                    x=x,
                    y=y,
                    accumulation_steps=accumulation_steps,
                    validation_data=validation_data,
                    callbacks=self._get_callbacks(),
                    **kwargs
                )
                histories.append(history)
                
            return histories
        
    def predict(
        self,
        x: List[np.ndarray],
        return_uncertainty: bool = True,
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes memory-efficient predictions with uncertainty estimation.
        """
        # Calculate number of batches
        total_samples = len(x[0])
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        predictions = []
        uncertainties = []
        
        # Predict in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            batch_x = [x_i[start_idx:end_idx] for x_i in x]
            
            batch_predictions = []
            batch_uncertainties = []
            
            for model in self.models:
                if self.uncertainty:
                    pred, unc = model.predict(batch_x, batch_size=batch_size)
                    batch_predictions.append(pred)
                    batch_uncertainties.append(unc)
                else:
                    pred = model.predict(batch_x, batch_size=batch_size)
                    batch_predictions.append(pred)
            
            predictions.append(np.mean(batch_predictions, axis=0))
            if self.uncertainty:
                uncertainties.append(
                    np.mean(batch_uncertainties, axis=0) + 
                    np.var(batch_predictions, axis=0)
                )
        
        # Concatenate batches
        final_predictions = np.concatenate(predictions, axis=0)
        
        if return_uncertainty and uncertainties:
            final_uncertainties = np.concatenate(uncertainties, axis=0)
            return final_predictions, final_uncertainties
            
        return final_predictions
        
    def _get_callbacks(self) -> List:
        """
        Creates a list of callbacks for training.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config['checkpoint_path'],
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config['log_dir'],
                histogram_freq=1
            )
        ]
        
        return callbacks
        
    def save_models(self, base_path: str):
        """
        Saves all models in the ensemble.
        """
        for i, model in enumerate(self.models):
            model.save(f"{base_path}/model_{i}.h5")
            
    def load_models(self, base_path: str, num_models: int):
        """
        Loads saved models into the ensemble.
        """
        self.models = [
            tf.keras.models.load_model(
                f"{base_path}/model_{i}.h5",
                custom_objects={
                    'SpatialAttentionLayer': SpatialAttentionLayer,
                    'TemporalAttentionLayer': TemporalAttentionLayer,
                    'SpatioTemporalAttention': SpatioTemporalAttention,
                    'UncertaintyLayer': UncertaintyLayer,
                    'ResidualConvLSTM2D': ResidualConvLSTM2D,
                    'FeatureFusion': FeatureFusion,
                    'WildfirePredictionLoss': WildfirePredictionLoss,
                    'UncertaintyLoss': UncertaintyLoss
                }
            )
            for i in range(num_models)
        ]
    
    def advanced_fit(
        self,
        x: List[np.ndarray],
        y: np.ndarray,
        validation_data: Optional[Tuple] = None,
        target_accuracy: float = 0.90,
        max_epochs: int = 1000,
        patience: int = 20,
        min_delta: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced training loop with adaptive learning and metrics tracking.
        """
        training_stats = {
            'total_epochs': 0,
            'best_accuracy': 0.0,
            'training_time': 0,
            'model_metrics': [],
            'learning_curves': {
                'accuracy': [],
                'loss': [],
                'val_accuracy': [],
                'val_loss': []
            }
        }
        
        start_time = time.time()
        best_weights = None
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):
            epoch_metrics = []
            
            # Train each ensemble member
            for i, model in enumerate(self.models):
                print(f"\nTraining model {i+1}/{self.num_ensemble} - Epoch {epoch+1}/{max_epochs}")
                
                # Train for one epoch
                history = model.fit(
                    x=x,
                    y=y,
                    validation_data=validation_data,
                    epochs=1,
                    verbose=1,
                    callbacks=self._get_callbacks(),
                    **kwargs
                )
                
                # Collect metrics
                metrics = {
                    'loss': history.history['loss'][-1],
                    'accuracy': history.history['accuracy'][-1]
                }
                
                if validation_data:
                    metrics.update({
                        'val_loss': history.history['val_loss'][-1],
                        'val_accuracy': history.history['val_accuracy'][-1]
                    })
                
                epoch_metrics.append(metrics)
            
            # Calculate ensemble metrics
            avg_metrics = self._calculate_ensemble_metrics(epoch_metrics)
            training_stats['model_metrics'].append(avg_metrics)
            
            # Update learning curves
            for key in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                if key in avg_metrics:
                    training_stats['learning_curves'][key].append(avg_metrics[key])
            
            # Check for improvement
            current_val_accuracy = avg_metrics.get('val_accuracy', avg_metrics['accuracy'])
            
            if current_val_accuracy > best_val_accuracy + min_delta:
                best_val_accuracy = current_val_accuracy
                epochs_without_improvement = 0
                best_weights = [model.get_weights() for model in self.models]
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Target accuracy check
            if current_val_accuracy >= target_accuracy:
                print(f"\nReached target accuracy of {target_accuracy}")
                break
            
            # Adjust learning rate if needed
            if epochs_without_improvement > patience // 2:
                self._adjust_learning_rate(0.5)
        
        # Restore best weights
        if best_weights:
            for model, weights in zip(self.models, best_weights):
                model.set_weights(weights)
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(x, y, validation_data)
        training_stats.update({
            'total_epochs': epoch + 1,
            'best_accuracy': best_val_accuracy,
            'training_time': time.time() - start_time,
            'final_metrics': final_metrics
        })
        
        # Save training stats
        self._save_training_stats(training_stats)
        
        return training_stats
    
    def _calculate_ensemble_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Calculate average metrics across ensemble"""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        return avg_metrics
    
    def _calculate_final_metrics(
        self,
        x: List[np.ndarray],
        y: np.ndarray,
        validation_data: Optional[Tuple]
    ) -> Dict:
        """Calculate comprehensive final metrics"""
        # Make predictions
        y_pred, uncertainties = self.predict(x, return_uncertainty=True)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.flatten(),
            y_pred_binary.flatten(),
            average='binary'
        )
        
        roc_auc = roc_auc_score(y.flatten(), y_pred.flatten())
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'mean_uncertainty': float(np.mean(uncertainties))
        }
        
        # Validation metrics if available
        if validation_data:
            val_pred, val_unc = self.predict(
                [validation_data[0][0], validation_data[0][1]],
                return_uncertainty=True
            )
            val_pred_binary = (val_pred > 0.5).astype(int)
            
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                validation_data[1].flatten(),
                val_pred_binary.flatten(),
                average='binary'
            )
            
            metrics.update({
                'val_precision': float(val_precision),
                'val_recall': float(val_recall),
                'val_f1_score': float(val_f1),
                'val_uncertainty': float(np.mean(val_unc))
            })
        
        return metrics
    
    def _adjust_learning_rate(self, factor: float):
        """Adjust learning rate for all models"""
        for model in self.models:
            current_lr = K.get_value(model.optimizer.learning_rate)
            new_lr = current_lr * factor
            K.set_value(model.optimizer.learning_rate, new_lr)
            print(f"\nReducing learning rate to {new_lr}")
    
    def _save_training_stats(self, stats: Dict):
        """Save training statistics to file"""
        save_path = Path(self.config['log_dir']) / 'training_stats.json'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def train_with_gradient_accumulation(
        self,
        x: List[np.ndarray],
        y: np.ndarray,
        accumulation_steps: int = 4,
        **kwargs
    ) -> Any:
        """Training step with gradient accumulation"""
        
        @tf.function
        def accumulate_gradients(x_batch, y_batch, model):
            # Initialize gradient accumulator
            accumulated_gradients = [
                tf.zeros_like(var) 
                for var in model.trainable_variables
            ]
            
            # Split batch for accumulation
            batch_size = tf.shape(x_batch[0])[0]
            split_size = batch_size // accumulation_steps
            
            total_loss = 0.0
            
            for i in range(accumulation_steps):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                
                x_micro = [x[start_idx:end_idx] for x in x_batch]
                y_micro = y_batch[start_idx:end_idx]
                
                with tf.GradientTape() as tape:
                    predictions = model(x_micro, training=True)
                    loss = self.loss_fn(y_micro, predictions)
                    scaled_loss = loss / accumulation_steps
                
                gradients = tape.gradient(
                    scaled_loss, 
                    model.trainable_variables
                )
                
                # Accumulate gradients
                accumulated_gradients = [
                    accum_grad + grad
                    for accum_grad, grad in zip(accumulated_gradients, gradients)
                ]
                
                total_loss += scaled_loss
                
            return accumulated_gradients, total_loss