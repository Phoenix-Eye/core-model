import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from typing import Tuple, Optional

class SpatialAttentionLayer(layers.Layer):
    """
    Spatial attention mechanism for focusing on relevant areas in spatial data.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionLayer, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid'
        )

    def call(self, inputs):
        # Compute average and max pooling along channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate pooled features
        pooled_features = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Generate attention map
        attention_map = self.conv(pooled_features)
        
        return inputs * attention_map

class TemporalAttentionLayer(layers.Layer):
    """
    Temporal attention mechanism for focusing on relevant time steps.
    """
    def __init__(self, units: int = 128):
        super(TemporalAttentionLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.V = self.add_weight(
            name="attention_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        # Calculate attention scores
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        
        # Apply attention weights
        context_vector = inputs * tf.expand_dims(attention_weights, -1)
        
        return context_vector, attention_weights

class SpatioTemporalAttention(layers.Layer):
    """
    Combined spatial and temporal attention for spatiotemporal data.
    """
    def __init__(self, spatial_kernel_size: int = 7, temporal_units: int = 128):
        super(SpatioTemporalAttention, self).__init__()
        self.spatial_attention = SpatialAttentionLayer(spatial_kernel_size)
        self.temporal_attention = TemporalAttentionLayer(temporal_units)

    def call(self, inputs):
        # Apply spatial attention first
        spatial_output = self.spatial_attention(inputs)
        
        # Reshape for temporal attention
        batch_size = tf.shape(spatial_output)[0]
        spatial_features = tf.reshape(spatial_output, [batch_size, -1, spatial_output.shape[-1]])
        
        # Apply temporal attention
        temporal_output, attention_weights = self.temporal_attention(spatial_features)
        
        # Reshape back to original format
        output = tf.reshape(temporal_output, tf.shape(inputs))
        
        return output, attention_weights

class UncertaintyLayer(layers.Layer):
    """
    Layer for estimating aleatoric uncertainty in predictions.
    """
    def __init__(self, units: int = 1):
        super(UncertaintyLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.dense_mean = layers.Dense(self.units)
        self.dense_var = layers.Dense(self.units, activation='softplus')

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        var = self.dense_var(inputs) + 1e-6  # Add small constant for numerical stability
        
        return mean, var

class ResidualConvLSTM2D(layers.Layer):
    """
    ConvLSTM2D layer with residual connections for better gradient flow.
    """
    def __init__(self, filters: int, kernel_size: Tuple[int, int], padding: str = 'same'):
        super(ResidualConvLSTM2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.convlstm = layers.ConvLSTM2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            return_sequences=True
        )
        
        # 1x1 convolution for matching dimensions if needed
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                padding='same'
            )
        else:
            self.projection = None

    def call(self, inputs):
        convlstm_output = self.convlstm(inputs)
        
        # Project input if necessary
        if self.projection is not None:
            residual = tf.keras.layers.TimeDistributed(self.projection)(inputs)
        else:
            residual = inputs
            
        return convlstm_output + residual

class FeatureFusion(layers.Layer):
    """
    Layer for fusing multiple feature streams with attention.
    """
    def __init__(self, output_dim: int):
        super(FeatureFusion, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("Input must be a list of tensors")
            
        self.attention_weights = [
            self.add_weight(
                name=f"fusion_weight_{i}",
                shape=(1,),
                initializer="ones",
                trainable=True
            )
            for i in range(len(input_shape))
        ]
        
        self.projection = layers.Dense(self.output_dim)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("Input must be a list of tensors")
            
        # Apply attention weights
        weighted_inputs = [
            inputs[i] * tf.nn.sigmoid(self.attention_weights[i])
            for i in range(len(inputs))
        ]
        
        # Concatenate weighted features
        concatenated = tf.concat(weighted_inputs, axis=-1)
        
        # Project to output dimension
        output = self.projection(concatenated)
        
        return output

class AdaptivePooling(layers.Layer):
    """
    Adaptive pooling layer for handling variable input sizes.
    """
    def __init__(self, output_size: Tuple[int, int], pooling_type: str = 'max'):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        input_height = tf.shape(inputs)[1]
        input_width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        
        stride_height = input_height // self.output_size[0]
        stride_width = input_width // self.output_size[1]
        
        kernel_size = [stride_height, stride_width]
        strides = [stride_height, stride_width]
        
        if self.pooling_type == 'max':
            pooling_layer = tf.keras.layers.MaxPool2D(
                pool_size=kernel_size,
                strides=strides,
                padding='valid'
            )
        else:
            pooling_layer = tf.keras.layers.AveragePooling2D(
                pool_size=kernel_size,
                strides=strides,
                padding='valid'
            )
            
        return pooling_layer(inputs)

def get_custom_objects():
    """
    Returns a dictionary of custom layer objects for model loading.
    """
    return {
        'SpatialAttentionLayer': SpatialAttentionLayer,
        'TemporalAttentionLayer': TemporalAttentionLayer,
        'SpatioTemporalAttention': SpatioTemporalAttention,
        'UncertaintyLayer': UncertaintyLayer,
        'ResidualConvLSTM2D': ResidualConvLSTM2D,
        'FeatureFusion': FeatureFusion,
        'AdaptivePooling': AdaptivePooling
    }