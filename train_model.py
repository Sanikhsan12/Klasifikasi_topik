import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_nlp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# * Load Dataset
def load_dataset(path):
    texts = []
    labels = []

    class_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(class_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        texts.append(f.read())
                        labels.append(class_name)
                except Exception as e:
                    print(f"Gagal membaca file {file_path}: {e}")
    return texts, labels

# * PLotting
def plot_train_history(history):
    plt.figure(figsize=(12, 5))
    # ? Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Akurasi (Training)')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Akurasi (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    plt.title('Akurasi Model')
    # ? Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss (Training)')
    plt.plot(history.history['val_loss'], label='Loss (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Model')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # ! Config
    dataset_path = 'bbc'
    model_preset = 'albert_base_en_uncased'
    batch_size = 32
    epochs = 10

    # * load Dataset
    texts, labels = load_dataset(dataset_path)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # * Config data
    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # * Modelisasi
    classifier = keras_nlp.models.AlbertClassifier.from_preset(
        model_preset,
        num_classes=num_classes
    )
    classifier.summary()

    # * Train
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    history = classifier.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # * Validasi hasil train
    val_pred = classifier.predict(x_val)
    val_pred_classes = np.argmax(val_pred, axis=-1)
    plot_train_history(history)
    plot_confusion_matrix(y_val, val_pred_classes, label_encoder.classes_)
    print(classification_report(y_val, val_pred_classes, target_names=label_encoder.classes_))

    # * prediksi hasil
    contoh_berita = [
        "Breaking news: AI is transforming the world.",
    ]
    # PERBAIKAN: Teruskan daftar string langsung
    prediksi = classifier.predict(contoh_berita)
    prediksi_kelas = np.argmax(prediksi, axis=-1)
    prediksi_label = label_encoder.inverse_transform(prediksi_kelas)

    for text, label in zip(contoh_berita, prediksi_label):
        print(f"Teks: {text}\nPrediksi Topik: {label.upper()}\n")

if __name__ == "__main__":
    main() 