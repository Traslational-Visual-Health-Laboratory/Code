import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# === FUNCION PARA GENERAR VENTANAS ===
def generar_ventanas(signal, fs=200, window_sec=25, step_sec=1):
    window_size = window_sec * fs
    step_size = step_sec * fs
    ventanas = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        ventanas.append(signal[start:start + window_size])
    return np.array(ventanas)  # shape: (n_ventanas, window_size)

# === BLOQUE RESIDUAL 1D ===
def residual_block_1d(x, filters, kernel_size=7, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# === MODELO COMPLETO RESNET + TRANSFORMER + MLP ===
def build_resnet_transformer_model(window_size=5000, n_windows=6, channels=1, embedding_dim=128, n_heads=4, transformer_layers=2):
    input_signal = layers.Input(shape=(n_windows, window_size, channels))

    # Aplicar ResNet a cada ventana
    def resnet_encoder(x):
        x = residual_block_1d(x, 64)
        x = residual_block_1d(x, 128, stride=2)
        x = layers.GlobalAveragePooling1D()(x)
        return x

    encoded_windows = layers.TimeDistributed(layers.Lambda(resnet_encoder))(input_signal)  # Shape: (batch, n_windows, features)

    # Embedding Posicional
    pos_encoding = tf.range(start=0, limit=n_windows, delta=1)
    pos_encoding = layers.Embedding(input_dim=n_windows, output_dim=embedding_dim)(pos_encoding)

    x = layers.Dense(embedding_dim)(encoded_windows)
    x = x + pos_encoding

    # Transformer Encoder Layers
    for _ in range(transformer_layers):
        attn_output = layers.MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        ffn = layers.Dense(embedding_dim, activation='relu')(x)
        ffn = layers.Dense(embedding_dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)

    # Global pooling across windows
    x = layers.GlobalAveragePooling1D()(x)

    # MLP final
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_signal, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    model = build_resnet_transformer_model()
    model.summary()
