import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from typing import List, Tuple, Dict, Optional
import numpy as np

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
                input_shape=self.config['input_shape'][1:]
            )
            # Freeze early layers
            for layer in base_model.layers[:100]:
                layer.trainable = False
        else:
            # Custom feature extraction if not using transfer learning
            inputs = Input(shape=self.config['input_shape'][1:])
            x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
            
            # Add residual blocks
            for filters in [64, 128, 256]:
                x = self._residual_block(x, filters)
            
            base_model = Model(inputs, x)
            
        return base_model
        
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
        Trains all models in the ensemble.
        """
        histories = []
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.num_ensemble}")
            history = model.fit(
                x=x,
                y=y,
                validation_data=validation_data,
                callbacks=self._get_callbacks(),
                **kwargs
            )
            histories.append(history.history)
            
        return histories
        
    def predict(
        self,
        x: List[np.ndarray],
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes predictions with uncertainty estimation using the ensemble.
        """
        predictions = []
        uncertainties = []
        
        # Get predictions from each model
        for model in self.models:
            if self.uncertainty:
                pred, unc = model.predict(x)
                predictions.append(pred)
                uncertainties.append(unc)
            else:
                pred = model.predict(x)
                predictions.append(pred)
                
        # Calculate ensemble statistics
        mean_prediction = np.mean(predictions, axis=0)
        
        if return_uncertainty:
            if self.uncertainty:
                # Combine model uncertainty and data uncertainty
                epistemic_uncertainty = np.var(predictions, axis=0)
                aleatoric_uncertainty = np.mean(uncertainties, axis=0)
                total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            else:
                # Use only ensemble variance as uncertainty
                total_uncertainty = np.var(predictions, axis=0)
                
            return mean_prediction, total_uncertainty
        
        return mean_prediction
        
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