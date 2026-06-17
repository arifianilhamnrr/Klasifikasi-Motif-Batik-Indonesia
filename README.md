# Klasifikasi Motif Batik Indonesia 🇮🇩
**Berdasarkan Pembelajaran Representasi Citra (Image Representation Learning)**

Proyek ini adalah implementasi *Deep Learning* menggunakan Convolutional Neural Network (CNN) berbasis **PyTorch** untuk mengklasifikasikan 20 jenis motif batik Indonesia. Dilengkapi dengan antarmuka web interaktif menggunakan **Gradio**.

## Fitur Utama
- 🔥 **Arsitektur Model**: Mendukung ResNet18/ResNet50 (*Transfer Learning*) dan arsitektur SimpleCNN custom.
- 🎨 **Web Interface**: Antarmuka unggah citra yang responsif, menampilkan probabilitas prediksi per kelas.
- 📊 **Dataset Tools**: Script otomatis untuk mengunduh, mengekstrak, dan memvalidasi proporsi dataset (*Stratified Train/Val Split*).
- 📓 **Jupyter Notebook**: Disediakan `Batik_Classifier_Notebook.ipynb` untuk eksplorasi dan riset.

---

## 📂 Struktur Repositori

```text
batik-cnn-classifier/
├── src/                  # Kode inti (Dataset, Model, Training, Inference)
├── app/gradio_app.py     # Source code Web App Gradio
├── scripts/              # Utilitas persiapan dataset
├── data/raw/             # Folder dataset mentah
├── data/processed/       # Folder dataset siap latih
├── checkpoints/          # Folder penyimpan model terbaik
├── requirements.txt      # Dependensi Python
└── Batik_Classifier_Notebook.ipynb
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Clone & Setup Environment
```bash
git clone https://github.com/arifianilhamnrr/Batik_Classifier.git
cd Batik_Classifier

# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Unduh Dataset
Contoh menggunakan public dataset Kaggle (pastikan Anda telah mengonfigurasi `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d hamdanialikhsan/batik-motifs -p data/raw --unzip
```
Atau jika via URL langsung:
```bash
python scripts/download_dataset.py --url "URL_DATASET" --out_dir data/raw --extract
```

### 3. Persiapan & Pembagian Dataset (Train/Val)
Script ini akan melakukan *stratified split* (otomatis membagi data menjadi Train & Val).
```bash
python scripts/prepare_dataset.py \
  --raw_dir data/raw/batik-motifs \
  --out_dir data/processed \
  --val_ratio 0.2 \
  --seed 42 \
  --overwrite

# Verifikasi hasil pemotongan
python scripts/verify_dataset.py --data_dir data/processed
```

---

## 🧠 Pelatihan Model (Training)

Jalankan perintah ini untuk melatih model menggunakan `resnet18`:
```bash
python src/train.py \
  --data_dir data/processed \
  --model_name resnet18 \
  --epochs 25 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_path checkpoints/batik_resnet18_best.pth
```
*(Model terbaik berdasarkan akurasi validasi akan otomatis tersimpan di folder `checkpoints/`).*

---

## 🌐 Menjalankan Web App (Inference)

Untuk menjalankan antarmuka web:
```bash
python app/gradio_app.py --port 7860
```
Buka browser dan akses ke `http://localhost:7860`.
> **Catatan:** Jika belum ada model yang dilatih, aplikasi otomatis membuat mode "Demo" dengan bobot acak.

---

## 📓 Notebook Eksperimen
Untuk kebutuhan riset atau pelaporan, buka `Batik_Classifier_Notebook.ipynb` melalui Jupyter.

```bash
jupyter lab Batik_Classifier_Notebook.ipynb
```

---
*Dibuat untuk keperluan riset Klasifikasi Motif Batik Indonesia.*