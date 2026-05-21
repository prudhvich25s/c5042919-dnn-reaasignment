import tensorflow as tf


def build_model(vocab_size, embedding_dim, max_length, lstm_units, dropout, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

        tf.keras.layers.LSTM(lstm_units, return_sequences=False),

        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model