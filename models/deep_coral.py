# models/deep_coral.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, AveragePooling1D, 
    Dropout, Flatten, Dense, Lambda
)
import numpy as np

def coral_loss(source, target):
    """
    Compute CORrelation ALignment (CORAL) loss to minimize the difference 
    in covariance between source and target feature distributions.

    Parameters:
        source (tf.Tensor): Feature matrix of the source (batch_size, features)
        target (tf.Tensor): Feature matrix of the target (batch_size, features)

    Returns:
        tf.Tensor: Scalar CORAL loss
    """
    d = tf.cast(tf.shape(source)[1], tf.float32)  # Feature dimensionality

    # Covariance of source and target
    source_cov = tf.matmul(
        tf.transpose(source), source
    ) / tf.cast(tf.shape(source)[0] - 1, tf.float32)
    target_cov = tf.matmul(
        tf.transpose(target), target
    ) / tf.cast(tf.shape(target)[0] - 1, tf.float32)

    # Frobenius norm of the difference
    loss = tf.reduce_mean(tf.square(source_cov - target_cov)) / (4.0 * d * d)
    return loss

def build_deep_coral_model(input_shape):
    """
    Build the Deep CORAL model with shared feature extraction layers.

    Parameters:
        input_shape (tuple): (time_steps, channels) or (features, 1), etc.

    Returns:
        model (tf.keras.Model): Keras model for Deep CORAL
    """
    # 1) Define Input
    input_layer = Input(shape=input_shape, name="input_layer")

    # 2) Example 1D Convolution + Batch Norm
    x = Conv1D(filters=8, kernel_size=3, padding="same", activation="relu", name="conv1")(input_layer)
    x = BatchNormalization(name="batch_norm1")(x)

    # 3) Square activation (custom Lambda)
    x = Lambda(lambda t: tf.math.square(t), name="squaring")(x)

    # 4) Pooling + Dropout
    x = AveragePooling1D(pool_size=2, name="pooling")(x)
    x = Dropout(0.3, name="dropout")(x)

    # 5) Flatten → Classification
    feature_layer = Flatten(name="feature_layer")(x)
    output_layer  = Dense(units=1, activation="sigmoid", name="output_layer")(feature_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name="DeepCORAL_Model")
    return model
