# 🚨**Project: Klasifikasi Serangan Jaringan Menggunakan Decision Tree**

## **Penjelasan**
Proyek ini bertujuan untuk mengklasifikasikan berbagai jenis serangan siber berdasarkan data jaringan menggunakan algoritma *Decision Tree Classifier*. Dataset berisi fitur-fitur lalu lintas jaringan dan label jenis serangan seperti:
- DDoS UDP Flood
- DoS ICMP Flood
- MITM ARP Spoofing
- MQTT DoS Publish Flood
- Recon Vulnerability Scan

## 📁Struktur 

### 1. **Import Library**
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
```

### 2. **Load Dataset**
Dataset dibaca dari Google Drive menggunakan `pd.read_csv()`:
```python
ar = pd.read_csv('/content/drive/MyDrive/Data_Mining/DDoS UDP Flood.csv')
ir = pd.read_csv('/content/drive/MyDrive/Data_Mining/DoS ICMP Flood.csv')
ur = pd.read_csv('/content/drive/MyDrive/Data_Mining/MITM ARP Spoofing.csv')
er = pd.read_csv("/content/drive/MyDrive/Data Mining /MQTT DoS Publish Flood.csv")
ro = pd.read_csv("/content/drive/MyDrive/Data Mining /Recon Vulnerability Scan.csv")
```

### 3. **Gabungkan Dataset**
Semua dataset digabung menggunakan `pd.concat()` dan hasilnya disimpan dalam `hasilgabung`.

### 4. **Pemisahan Data**
Data dibagi menjadi:
- **Fitur (X)**: `A` – semua kolom kecuali label.
- **Label (y)**: `B` – kolom `Attack Name`.

Kemudian dilakukan *split* untuk pelatihan dan pengujian:
```python
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42)
```

### 5. **Training Model**
Menggunakan `DecisionTreeClassifier`:
```python
santi = DecisionTreeClassifier(criterion='entropy', splitter='random')
santi.fit(A_train, B_train)
```

### 6. **Prediksi dan Evaluasi**
Melakukan prediksi dan menghitung akurasi:
```python
B_pred = santi.predict(A_test)
accuracy = accuracy_score(B_test, B_pred)
```

### 7. **Visualisasi**
- **Pohon Keputusan** divisualisasikan dengan `plot_tree()`.
- **Confusion Matrix** divisualisasikan dengan `seaborn.heatmap()`.

---

### 🚨Hasil Evaluasi
- **Akurasi Model**: ~86%






