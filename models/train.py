import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from models.tacnn import (
    multi_band_filter, 
    apply_csp_per_band,
    build_tacnn_model
)
from models.evaluate import evaluate_model
from sklearn.metrics import accuracy_score


def train_tacnn_pipeline(
    X_train, y_train,
    X_val, y_val,
    frequency_bands=[(4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)],
    n_components=6,
    epochs=100,
    batch_size=16,
    patience=5
):
    """
    Complete TA-CSPNN pipeline on the training + validation set.

    Returns:
        model       - Trained TA-CSPNN Keras model
        filters     - CSP filters used for transformation
        freq_bands  - Frequency bands used for multi-band filtering
    """

    # ------------------------------------------------
    # 1) Multi-band filter (Train + Val)
    # ------------------------------------------------
    X_train_filt = multi_band_filter(X_train, frequency_bands)
    X_val_filt   = multi_band_filter(X_val,   frequency_bands)

    # ------------------------------------------------
    # 2) Fit CSP filters on training set only
    # ------------------------------------------------
    X_train_csp, X_val_csp, filters = apply_csp_per_band(X_train_filt, y_train, X_val_filt, frequency_bands, n_components)

    # ------------------------------------------------
    # 3) Change -1/1 to 0/1 for deep learning
    # ------------------------------------------------
    y_train_new = np.where(y_train == -1, 0, 1)
    y_val_new   = np.where(y_val == -1, 0, 1)

    # Reshape for 1D CNN => (samples, features, 1)
    X_train_csp = X_train_csp.reshape((X_train_csp.shape[0], X_train_csp.shape[1], 1))
    X_val_csp   = X_val_csp.reshape((X_val_csp.shape[0], X_val_csp.shape[1], 1))

    # ------------------------------------------------
    # 4) Build TA-CSPNN model
    # ------------------------------------------------
    model = build_tacnn_model(input_shape=(X_train_csp.shape[1], 1))
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    # ------------------------------------------------
    # 5) Train + Evaluate on Validation
    # ------------------------------------------------
    model.fit(
        X_train_csp, y_train_new,
        validation_data=(X_val_csp, y_val_new),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on validation set
    print("\n=== TA-CSPNN Validation Evaluation ===")
    X_val_csp = X_val_csp.reshape((X_val_csp.shape[0], X_val_csp.shape[1], 1))
    probs = model.predict(X_val_csp)
    preds = 2 * (probs >= 0.5) - 1

    acc = accuracy_score(y_val, preds)
    print("Validation Accuracy:", acc)

    return model, filters, frequency_bands



def coral_loss(source_features, target_features):
    """Compute CORAL loss between source and target features."""
    # Center the features
    source_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
    target_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
    source_centered = source_features - source_mean
    target_centered = target_features - target_mean

    # Compute covariance matrices
    batch_size = tf.cast(tf.shape(source_features)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(target_features)[0], tf.float32)
    
    source_cov = (tf.matmul(source_centered, source_centered, transpose_a=True) / 
                  (batch_size - 1))
    target_cov = (tf.matmul(target_centered, target_centered, transpose_a=True) / 
                  (target_batch_size - 1))

    # Compute Frobenius norm
    coral_loss = tf.reduce_sum(tf.square(source_cov - target_cov))
    return coral_loss


def train_deep_coral_model(
    model,
    X_train, Y_train,
    X_test, Y_test,
    epochs=50,
    batch_size=16,
    alpha=0.01,
    patience=5,
    verbose=1
):
    """
    Custom training loop for Deep CORAL, using a shared feature extractor
    and CORAL loss to align distributions of source (train) & target (test).

    Parameters:
        model (tf.keras.Model): Deep CORAL model (built via build_deep_coral_model).
        X_train, Y_train: Source data and labels
        X_test, Y_test:  Target data and labels (unlabeled or partially labeled)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for source data
        alpha (float): Weight factor for the CORAL loss
        verbose (int): Print frequency. 0 = silent, 1 = print each epoch.

    Returns:
        final_accuracy (float): Accuracy on target domain after training
        model (tf.keras.Model): Trained model
        feature_extractor (tf.keras.Model): Model subgraph that outputs features
    """
    # 1) Extract feature representation sub-model
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("feature_layer").output)

    # 2) Define optimizer & classification loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    classification_loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Convert data to tf.Tensor if needed
    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    Y_train_tf = tf.constant(Y_train, dtype=tf.float32)
    X_test_tf  = tf.constant(X_test,  dtype=tf.float32)
    Y_test_tf  = tf.constant(Y_test,  dtype=tf.float32)

    train_size = X_train.shape[0]
    n_batches = int(np.ceil(train_size / batch_size))

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Create TensorFlow dataset and shuffle
        dataset = tf.data.Dataset.from_tensor_slices((X_train_tf, Y_train_tf))
        dataset = dataset.shuffle(buffer_size=train_size).batch(batch_size)

        total_loss_val = 0.0

        for X_batch_source, Y_batch_source in dataset:

            with tf.GradientTape() as tape:
                # Forward pass source
                source_preds = model(X_batch_source, training=True)
                # Forward pass target
                target_preds = model(X_test_tf, training=True)

                # Extract features
                source_features = feature_extractor(X_batch_source, training=True)
                target_features = feature_extractor(X_test_tf, training=True)

                # Classification loss only for source domain
                loss_classification = classification_loss_fn(Y_batch_source, source_preds)
                # CORAL loss to align source & target
                loss_coral = coral_loss(source_features, target_features)

                # Combine losses
                total_loss = loss_classification + alpha * loss_coral

            # Backprop
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss_val += total_loss.numpy()

        avg_loss = total_loss_val / n_batches

        val_preds = model(X_test_tf, training=False)
        val_loss = classification_loss_fn(Y_test_tf, val_preds).numpy()

        if verbose == 1:
            # Evaluate classification on source, or skip if unlabeled
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience
            best_model_weights = model.get_weights()  # Save best model state
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            model.set_weights(best_model_weights)  # Restore best model
            break

        
    # Final evaluation on target domain
    preds_test = model(X_test_tf, training=False).numpy()
    preds_test_bin = (preds_test >= 0.5).astype(int).ravel()
    final_predictions = 2 * preds_test_bin - 1
    final_accuracy = np.mean(final_predictions.flatten() == Y_test)  # Compare to original labels (-1,1)

    if verbose == 1:
        print(f"Deep CORAL Adaptation Test Accuracy: {final_accuracy:.4f}")

    return final_accuracy, model, feature_extractor