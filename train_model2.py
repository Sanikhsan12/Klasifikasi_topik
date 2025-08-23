import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

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

# * Plotting 
def plot_train_history(history):
    plt.figure(figsize=(12, 5))
    # ? Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Akurasi (Training)')
    plt.plot(history['val_acc'], label='Akurasi (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    plt.title('Akurasi Model')
    # ? Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss (Training)')
    plt.plot(history['val_loss'], label='Loss (Validation)')
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

# * Encoding text
def encoded_texts(tokenizer, texts, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return (
        torch.cat(input_ids, dim=0),
        torch.cat(attention_masks, dim=0)
    )

# * Main PRG
def main(contoh_berita):
    # ! Config
    dataset_path = 'bbc' 
    model_name = 'albert-base-v2'
    batch_size = 32
    epochs = 10
    max_len = 128

    # * Load Dataset
    texts, labels = load_dataset(dataset_path) 

    # * Label encoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # * Config data
    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # * Tokenisasi
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    train_inputs, train_masks = encoded_texts(tokenizer, x_train, max_len)
    val_inputs, val_masks = encoded_texts(tokenizer, x_val, max_len)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # * Data Loader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


    # * modelisasi
    model = AlbertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # * Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # * Train
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train_preds = 0
        
        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            
            model.zero_grad()
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=1).flatten()
            correct_train_preds += torch.sum(preds == b_labels)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_acc = correct_train_preds.double() / len(train_data)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # * Validasi
        model.eval()
        total_val_loss = 0
        correct_val_preds = 0
        all_preds = []
        all_labels = []

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            loss = outputs.loss
            logits = outputs.logits
            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            correct_val_preds += torch.sum(preds == b_labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc = correct_val_preds.double() / len(val_data)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {avg_val_loss}, Val Acc: {val_acc}")

    # * Visualisasi
    plot_train_history(history)
    plot_confusion_matrix(all_labels, all_preds, label_encoder.classes_)

    # * Contoh Prediksi
    encoded_contoh = tokenizer.batch_encode_plus(
        contoh_berita,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_contoh['input_ids'].to(device)
    attention_mask = encoded_contoh['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    logits = outputs.logits
    
    
    # * hasil
    prediksi_kelas = torch.argmax(logits, dim=1).cpu().numpy()
    prediksi_label = label_encoder.inverse_transform(prediksi_kelas)
    for text, label in zip(contoh_berita, prediksi_label):
        print(f"Berita: {text}\nPrediksi: {label}\n")


if __name__ == "__main__":

    contoh_berita = [
        "Breaking news: AI is transforming the world."
    ]
    main(contoh_berita=contoh_berita)