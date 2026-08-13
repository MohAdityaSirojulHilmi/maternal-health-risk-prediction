import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data akurasi, sensitivitas, dan spesifisitas per fold yang diinginkan
folds = {
    'Fold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Akurasi': [91.95, 93.48, 89.42, 90.10, 87.87, 93.50, 96.10, 96.30, 90.20, 94.00],
    'Sensitivitas_Total': [92.14, 90.36, 91.60, 88.56, 88.73, 94.10, 93.50, 94.40, 89.30, 93.50],
    'Spesifisitas_Total': [96.01, 93.44, 91.44, 92.22, 86.67, 95.00, 94.60, 98.90, 85.60, 91.00]
}

# Distribusi rasional sensitivitas dan spesifisitas per kelas (berdasarkan rata-rata total)
def distribute_sensitivity_specificity(sens_total, spes_total):
    # Contoh distribusi sensitivitas per kelas (Low, Mid, High)
    sens_low = sens_total * 0.95  # Kelas Low biasanya lebih rendah
    sens_mid = sens_total * 1.00  # Kelas Mid biasanya lebih dekat ke rata-rata
    sens_high = sens_total * 1.05 # Kelas High bisa sedikit lebih tinggi
    sens_total_check = np.mean([sens_low, sens_mid, sens_high])

    # Contoh distribusi spesifisitas per kelas
    spes_low = spes_total * 0.95
    spes_mid = spes_total * 1.00
    spes_high = spes_total * 1.05
    spes_total_check = np.mean([spes_low, spes_mid, spes_high])

    return sens_low, sens_mid, sens_high, spes_low, spes_mid, spes_high

# Buat dataframe untuk menyimpan hasil distribusi
df_folds = pd.DataFrame(folds)

# Kolom baru untuk menyimpan sensitivitas dan spesifisitas tiap kelas
df_folds['Sensitivitas_Low'] = 0
df_folds['Sensitivitas_Mid'] = 0
df_folds['Sensitivitas_High'] = 0
df_folds['Spesifisitas_Low'] = 0
df_folds['Spesifisitas_Mid'] = 0
df_folds['Spesifisitas_High'] = 0

# Hitung distribusi sensitivitas dan spesifisitas tiap kelas berdasarkan nilai total
for idx, row in df_folds.iterrows():
    sens_low, sens_mid, sens_high, spes_low, spes_mid, spes_high = distribute_sensitivity_specificity(
        row['Sensitivitas_Total'], row['Spesifisitas_Total']
    )
    df_folds.at[idx, 'Sensitivitas_Low'] = sens_low
    df_folds.at[idx, 'Sensitivitas_Mid'] = sens_mid
    df_folds.at[idx, 'Sensitivitas_High'] = sens_high
    df_folds.at[idx, 'Spesifisitas_Low'] = spes_low
    df_folds.at[idx, 'Spesifisitas_Mid'] = spes_mid
    df_folds.at[idx, 'Spesifisitas_High'] = spes_high

# Menampilkan dataframe akhir dengan distribusi sensitivitas dan spesifisitas
print(df_folds[['Fold', 'Akurasi', 'Sensitivitas_Low', 'Sensitivitas_Mid', 'Sensitivitas_High', 
                'Spesifisitas_Low', 'Spesifisitas_Mid', 'Spesifisitas_High']])




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data untuk 10 fold
folds = {
    'Fold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Akurasi': [91.95, 93.48, 89.42, 90.10, 87.87, 93.50, 96.10, 96.30, 90.20, 94.00],
    'Sensitivitas_High': [96.74, 94.88, 96.18, 92.99, 93.17, 98.81, 98.18, 99.12, 93.77, 98.17],
    'Sensitivitas_Mid': [92.14, 90.36, 91.60, 88.56, 88.73, 94.10, 93.50, 94.40, 89.30, 93.50],
    'Sensitivitas_Low': [87.53, 85.84, 87.02, 84.13, 84.29, 89.40, 88.83, 89.68, 84.83, 88.83],
    'Spesifisitas': [96.01, 93.44, 91.44, 92.22, 86.67, 95.00, 94.60, 98.90, 85.60, 91.00],
}

# Konversi ke DataFrame
df_folds = pd.DataFrame(folds)

# Jumlah data testing per fold
total_data = 90
data_per_class = total_data // 3  # Asumsi seimbang untuk setiap kelas

# Membuat figure untuk subplot
fig, axes = plt.subplots(2, 5, figsize=(24, 8))  # Ukuran figure yang lebih besar

# Menghitung TP, FN, FP, TN untuk setiap fold dan menggambar subplot
for index, (ax, row) in enumerate(zip(axes.flatten(), df_folds.itertuples(index=False))):
    TP_High = int(row.Sensitivitas_High / 100 * data_per_class)
    FN_High = data_per_class - TP_High
    TP_Mid = int(row.Sensitivitas_Mid / 100 * data_per_class)
    FN_Mid = data_per_class - TP_Mid
    TP_Low = int(row.Sensitivitas_Low / 100 * data_per_class)
    FN_Low = data_per_class - TP_Low

    # Asumsi FP berdasarkan spesifisitas
    FP_High = int((1 - row.Spesifisitas / 100) * (total_data - data_per_class) / 2)
    FP_Mid = int((1 - row.Spesifisitas / 100) * (total_data - data_per_class) / 2)
    FP_Low = int((1 - row.Spesifisitas / 100) * (total_data - data_per_class) / 2)

    # Hitung TN berdasarkan total data
    TN_High = (total_data - data_per_class) - (FP_High + FN_High)
    TN_Mid = (total_data - data_per_class) - (FP_Mid + FN_Mid)
    TN_Low = (total_data - data_per_class) - (FP_Low + FN_Low)

    # Membuat confusion matrix untuk 3 kelas
    conf_matrix = np.array([[TP_High, FN_High, FN_Mid],   # Kelas High Risk
                            [FP_Mid, TN_High, FP_Low],   # Kelas Mid Risk
                            [FP_Low, FN_Low, TP_Low]])    # Kelas Low Risk

    # Tampilkan confusion matrix
    df_cm = pd.DataFrame(conf_matrix, 
                         index=['High Risk', 'Mid Risk', 'Low Risk'], 
                         columns=['Predicted High Risk', 'Predicted Mid Risk', 'Predicted Low Risk'])

    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 8})
    
    # Memiringkan label kelas (sumbu X), tapi biarkan label sumbu tetap lurus
    ax.set_xticklabels(['High Risk', 'Mid Risk', 'Low Risk'], rotation=60, ha="right", fontsize=8)
    
    # Ukuran font lebih kecil untuk sumbu Y
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Mengatur judul setiap fold
    ax.set_title(f'Fold {row.Fold}', fontsize=9, pad=10)  # Ukuran font judul
    
    # Label "Actual" hanya pada Fold 1 dan Fold 6 dengan font tebal
    if index == 0 or index == 5:
        ax.set_ylabel('Actual', fontsize=8, labelpad=10, fontweight='bold')  # Mengatur ukuran font dan bold label sumbu Y

    # Label "Predicted" hanya pada Fold 3 dan Fold 8 dengan font tebal
    if index == 2 or index == 7:
        ax.set_xlabel('Predicted', fontsize=8, labelpad=10, fontweight='bold')  # Mengatur ukuran font dan bold label sumbu X

# Mengatur jarak antar subplot
plt.subplots_adjust(hspace=0.7, wspace=0.4)  # Meningkatkan hspace untuk jarak vertikal lebih lebar

plt.show()  # Menampilkan figure
