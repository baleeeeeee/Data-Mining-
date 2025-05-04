# 📚 Ringkasan Materi Pra-UTS

Ringkasan ini mencakup penggunaan **pandas**, **matplotlib**, serta algoritma **Naive Bayes** dan **Decision Tree** untuk analisis dan klasifikasi data.

---

## 1. Pandas

### 📌 Tujuan
Digunakan untuk manipulasi dan analisis data.

### 🔧 Fitur Utama
- Struktur data `DataFrame` dan `Series`.
- Fungsi penting:
  - `read_csv()`, `groupby()`, `merge()`, `pivot_table()`.

### 💡 Contoh Penggunaan
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
```

---

## 2. Matplotlib

### 📌 Tujuan
Visualisasi data dalam bentuk grafik.

### 🔧 Fitur Utama
- Fungsi visualisasi: `plot()`, `bar()`, `hist()`, `scatter()`.
- Kustomisasi grafik: `xlabel()`, `ylabel()`, `title()`, `legend()`.

### 💡 Contoh Penggunaan
```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Grafik Garis')
plt.xlabel('Sumbu X')
plt.ylabel('Sumbu Y')
plt.show()
```

---

## 3. Naive Bayes

### 📌 Tujuan
Klasifikasi data berbasis probabilitas.

### 🔧 Karakteristik
- Menggunakan Teorema Bayes.
- Asumsi independensi antar fitur.
- Cepat dan efisien untuk dataset besar.

### 💡 Contoh Penggunaan
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 4. Decision Tree

### 📌 Tujuan
Membuat model prediktif berbentuk pohon keputusan.

### 🔧 Karakteristik
- Mudah diinterpretasikan dan divisualisasikan.
- Menangani fitur numerik dan kategorikal.

### 💡 Contoh Penggunaan
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📎 Contoh Program di Google Colab
[🔗 Buka di Google Colab](https://colab.research.google.com/drive/1IIlMykBL0ltWwj-dXxyZ1Ylw-NP7w9oe)
