import yaml
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import load_data, preprocess, print_distribution
from model import build_model


# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
texts, labels = load_data()

print("Dataset size:", len(texts))
print_distribution(labels)

# Preprocess
X_train, X_val, y_train, y_val, tokenizer = preprocess(
    texts,
    labels,
    config["data"]["vocab_size"],
    config["data"]["max_length"]
)

# Build model
model = build_model(
    config["data"]["vocab_size"],
    config["model"]["embedding_dim"],
    config["data"]["max_length"],
    config["model"]["lstm_units"],
    config["model"]["dropout"],
    config["model"]["num_classes"]
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["training"]["epochs"],
    batch_size=config["training"]["batch_size"]
)

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# Plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.savefig("results/figures/loss_curve.png")
plt.show()