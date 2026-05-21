import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data():
    dataset = load_dataset("daniel3303/StoryReasoning", split="train")

    texts = []
    labels = []

    for item in dataset:
        story = item["story"]

        for i, sentence in enumerate(story[:5]):
            texts.append(sentence)
            labels.append(i)  # 0–4

    return texts, np.array(labels)


def preprocess(texts, labels, vocab_size, max_length):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')

    X_train, X_val, y_train, y_val = train_test_split(
        padded, labels, test_size=0.2, stratify=labels
    )

    return X_train, X_val, y_train, y_val, tokenizer


def print_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u+1}: {c}")