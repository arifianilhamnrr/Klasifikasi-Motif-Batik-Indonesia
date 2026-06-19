# Klasifikasi Motif Batik Indonesia

Proyek klasifikasi citra berbasis **PyTorch** untuk mengenali **20 motif batik Indonesia** menggunakan CNN (*Transfer Learning* ResNet18/ResNet50). Dilengkapi web app **Gradio**, bot **Telegram**, dan pipeline dataset dari **beberapa sumber publik** yang digabung menjadi satu dataset latih.

**Repository:** [github.com/arifianilhamnrr/Klasifikasi-Motif-Batik-Indonesia](https://github.com/arifianilhamnrr/Klasifikasi-Motif-Batik-Indonesia)

---

## Fitur Utama

- **20 kelas motif batik** — dataset digabung dari banyak sumber, bukan satu dataset saja
- **Model siap pakai** — checkpoint hasil training (~83% val accuracy) tersedia via Git LFS
- **Arsitektur fleksibel** — ResNet18, ResNet50 (*transfer learning*), atau SimpleCNN custom
- **Web app Gradio** — upload gambar, lihat prediksi + probabilitas per kelas
- **Telegram bot** — kirim foto batik, dapat prediksi langsung di chat
- **Pipeline dataset** — script unduh, ekstrak, split train/val, merge multi-sumber, verifikasi
- **Jupyter Notebook** — `Batik_Classifier_Notebook.ipynb` untuk eksperimen

---

## Model & Performa

| Checkpoint | Ukuran | Val Accuracy | Keterangan |
|---|---|---|---|
| `checkpoints/batik_best.pth` | ~129 MB | **~83%** | Model utama (50 epoch) |
| `checkpoints/batik_resnet18_best.pth` | ~107 MB | ~74.5% | Model alternatif (29 epoch) |

- **Arsitektur:** ResNet18 (pretrained ImageNet)
- **Kelas:** 20 motif
- **Data latih:** 4.175 gambar train / 1.067 gambar val
- **Input:** 224×224 RGB, dinormalisasi ImageNet

Checkpoint disimpan di GitHub menggunakan **Git LFS** — ikut terunduh saat clone (butuh `git lfs` terpasang).

---

## Quick Start — Langsung Pakai Model

```bash
# 1. Install Git LFS (sekali saja)
git lfs install

# 2. Clone repository
git clone https://github.com/arifianilhamnrr/Klasifikasi-Motif-Batik-Indonesia.git
cd Klasifikasi-Motif-Batik-Indonesia

# 3. Pastikan checkpoint terunduh
git lfs pull

# 4. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Jalankan web app
python app/gradio_app.py --port 7860
```

Buka `http://localhost:7860` → upload gambar batik → dapat prediksi langsung. Tidak perlu download dataset atau training ulang.

---

## 20 Kelas Motif

| # | Kelas | Train | Val |
|---|---|---|---|
| 1 | batik-bali | 159 | 41 |
| 2 | batik-betawi | 153 | 39 |
| 3 | batik-celup | 481 | 122 |
| 4 | batik-cendrawasih | 207 | 53 |
| 5 | batik-ceplok | 107 | 28 |
| 6 | batik-ciamis | 65 | 17 |
| 7 | batik-garutan | 59 | 15 |
| 8 | batik-gentongan | 197 | 51 |
| 9 | batik-kawung | 449 | 114 |
| 10 | batik-keraton | 253 | 64 |
| 11 | batik-lasem | 123 | 32 |
| 12 | batik-megamendung | 683 | 171 |
| 13 | batik-parang | 318 | 82 |
| 14 | batik-pekalongan | 72 | 18 |
| 15 | batik-priangan | 89 | 24 |
| 16 | batik-sekar | 153 | 39 |
| 17 | batik-sidoluhur | 175 | 46 |
| 18 | batik-sidomukti | 180 | 46 |
| 19 | batik-sogan | 93 | 24 |
| 20 | batik-tambal | 159 | 41 |

---

## Sumber Dataset

Dataset final **bukan dari satu sumber**, melainkan hasil penggabungan beberapa dataset publik batik Indonesia. Setiap sumber menyumbang kelas atau menambah variasi citra pada kelas yang sudah ada.

### Ringkasan sumber

| # | Sumber | Link | Lokasi / File | Kontribusi |
|---|---|---|---|---|
| 1 | **Batik Motifs** (Kaggle) | [hamdanialikhsan/batik-motifs](https://www.kaggle.com/datasets/hamdanialikhsan/batik-motifs) | `indonesian-batik-motifs.zip` | 630 images, 20 classes (dataset awal) |
| 2 | **Batik Nusantara** (Kaggle) | [dionisiusdh/indonesian-batik-motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) | `batik-nusantara-batik-indonesia-dataset.zip` | 983 images, 20 classes — Kelas regional: Bali, Betawi, Dayak, Cendrawasih, Kawung, Parang, Megamendung, Keraton, dll. |
| 3 | **Indonesian Batik Enhanced & Cleaned** (Kaggle) | [fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned](https://www.kaggle.com/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned) | `indonesian-batik-dataset-enhanced-and-cleaned.zip` | 1089 images — Ceplok, Sogan, Sidoluhur, Sekar, Tambal, Lasem, Priangan, Cendrawasih, Megamendung, Parang, Kawung, Celup |
| 4 | **Roboflow Batik** | [universe.roboflow.com/a-co6il/batik-tsrj7](https://universe.roboflow.com/a-co6il/batik-tsrj7) | `batikimages/` | ~4.357 citra (CC BY 4.0): Insang, Kawung, Parang, Megamendung, Sidoluhur, Truntum, Tumpal |
| 5 | **Corak App DATASETv7** (Kaggle) | [alfanme/indonesian-batik-motifs-corak-app](https://www.kaggle.com/datasets/alfanme/indonesian-batik-motifs-corak-app) | `corak-app/DATASETv7/TRAIN/` | 210 images — Batik Cendrawasih (70), Batik Sekar Jagad (70), Batik Tambal (70) |
| 6 | **3 Dataset Batik** (Kaggle) | [laoderhizwanyusuf/dataset-batik-dayak-batik-betawi-dan-batik-bali](https://www.kaggle.com/datasets/laoderhizwanyusuf/dataset-batik-dayak-batik-betawi-dan-batik-bali) | `3 Dataset Batik/` | 303 images — Batik Bali (101), Batik Betawi (100), Batik Dayak (101) |
| 7 | **BatikSnap Dataset** (Kaggle) | [syahdanputra/batiksnap-dataset](https://www.kaggle.com/datasets/syahdanputra/batiksnap-dataset) | `batiksnap-dataset.zip` | 1436 images — Citra motif klasik (Insang, Parang, Kawung, Mega Mendung, Sidoluhur, Truntum, Tumpal) |
| 8 | **Google Drive Upload** | (dataset pribadi) | Google Drive | 2 files, 70.8 MB total |

> **Catatan:** File dataset mentah (`data/raw/`, `data/processed/`) **tidak** disertakan di repository karena ukurannya besar. Yang di-push hanya kode, script, dan checkpoint model via Git LFS.

### Alur penggabungan dataset

```
┌─────────────────────────────────────────────────────────────┐
│  Sumber 1–8  →  data/raw/  (folder per kelas / per sumber)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
  prepare_dataset.py              merge_datasets.py
  (stratified train/val split)    (gabung dataset1 + dataset2
          │                        ke kelas yang sudah ada)
          └────────────────┬────────────────┘
                           ▼
                  data/processed/
                  ├── train/<kelas>/*.jpg
                  └── val/<kelas>/*.jpg
                           │
                           ▼
                    src/train.py
                           │
                           ▼
              checkpoints/batik_best.pth
```

### Mapping kelas tambahan

Script `src/merge_datasets.py` menggabungkan dataset berformat berbeda ke skema penamaan 20 kelas:

| Nama di dataset sumber | Nama kelas final | Sumber |
|---|---|---|
| `Gentongan` | `batik-gentongan` | Enhanced & Cleaned |
| `IkatCelup` | `batik-celup` | Enhanced & Cleaned |
| `Kawung` | `batik-kawung` | Enhanced & Cleaned + BatikSnap |
| `Megamendung` / `Jawa_Barat_Megamendung` | `batik-megamendung` | Enhanced & Cleaned |
| `Parang` / `Solo_Parang` / `Yogyakarta_Parang` | `batik-parang` | Enhanced & Cleaned |
| `sidomukti` | `batik-sidomukti` | Google Drive + BatikSnap |
| `Ceplok` | `batik-ceplok` | Enhanced & Cleaned |
| `Sogan` | `batik-sogan` | Enhanced & Cleaned |
| `Sidoluhur` | `batik-sidoluhur` | Enhanced & Cleaned + BatikSnap |
| `Sekar` | `batik-sekar` | Enhanced & Cleaned |
| `Tambal` | `batik-tambal` | Enhanced & Cleaned + Corak App |
| `Lasem` | `batik-lasem` | Enhanced & Cleaned |
| `Priangan_Merak_Ngibing` | `batik-priangan` | Enhanced & Cleaned |
| `Papua_Cendrawasih` | `batik-cendrawasih` | Enhanced & Cleaned |
| `Batik Cendrawasih` | `batik-cendrawasih` | Corak App DATASETv7 |
| `Batik Sekar Jagad` | `batik-sekar` | Corak App DATASETv7 |
| `Batik Tambal` | `batik-tambal` | Corak App DATASETv7 |
| `Batik Bali` | `batik-bali` | 3 Dataset Batik |
| `Batik Betawi` | `batik-betawi` | 3 Dataset Batik |

---

## Panduan Lengkap: Unduh & Siapkan Dataset

### 1. Clone & setup environment

```bash
git lfs install
git clone https://github.com/arifianilhamnrr/Klasifikasi-Motif-Batik-Indonesia.git
cd Klasifikasi-Motif-Batik-Indonesia

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Unduh dataset dari berbagai sumber

Letakkan semua file ke `data/raw/`. Contoh:

**Kaggle — Batik Motifs:**
```bash
pip install kaggle
kaggle datasets download -d hamdanialikhsan/batik-motifs -p data/raw --unzip
```

**Kaggle — Batik Nusantara:**
```bash
# Cari dataset batik nusantara / batik indonesia di Kaggle, lalu:
kaggle datasets download -d <username>/<dataset-slug> -p data/raw --unzip
```

**Roboflow — Batik Dataset:**
- Buka [universe.roboflow.com/a-co6il/batik-tsrj7](https://universe.roboflow.com/a-co6il/batik-tsrj7)
- Export format **Folder per Class** → ekstrak ke `data/raw/batikimages/`

**Dataset lainnya** — unduh manual atau via URL:
```bash
python scripts/download_dataset.py --url "URL_ZIP_DATASET" --out_dir data/raw --extract
```

Sumber yang perlu diunduh secara terpisah:
- `indonesian-batik-dataset-enhanced-and-cleaned.zip` → ekstrak ke `data/raw/dataset1_extracted/`
- `dataset2.zip` → ekstrak ke `data/raw/dataset2_extracted/`
- `dataset-batik-dayak-batik-betawi-dan-batik-bali.zip`
- `batiksnap-dataset.zip`
- Folder `3 Dataset Batik/` dan `corak-app/`

### 3. Normalisasi & split train/val

Pastikan struktur folder mentah mengikuti format `data/raw/<nama-kelas>/*.jpg`, lalu jalankan:

```bash
python scripts/prepare_dataset.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --val_ratio 0.2 \
  --seed 42 \
  --overwrite

python scripts/verify_dataset.py --data_dir data/processed
```

### 4. (Opsional) Merge dataset tambahan

Jika sudah punya `dataset1_extracted/` dan `dataset2_extracted/`:

```bash
python src/merge_datasets.py
```

Script ini menambahkan citra dari dataset enhanced & sidomukti ke `data/processed/` yang sudah ada.

---

## Pelatihan Model

```bash
python src/train.py \
  --data_dir data/processed \
  --model_name resnet18 \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_path checkpoints/batik_best.pth
```

Model terbaik (berdasarkan akurasi validasi) otomatis disimpan ke `--save_path`.

---

## Inference

### Web App (Gradio)

```bash
python app/gradio_app.py --port 7860
```

Gradio otomatis memuat `checkpoints/batik_best.pth`. Jika checkpoint tidak ditemukan, app berjalan dalam **demo mode** (bobot acak, prediksi tidak akurat).

### Telegram Bot

```bash
pip install aiogram
export TELEGRAM_BOT_TOKEN="token-bot-telegram-anda"
python telegram_bot.py --checkpoint checkpoints/batik_best.pth
```

Kirim foto batik ke bot → dapat prediksi motif + confidence.

### Python API

```python
from PIL import Image
from src.inference import load_model_and_meta, predict_batik

model, classes = load_model_and_meta("checkpoints/batik_best.pth")
result = predict_batik(Image.open("gambar.jpg"), model, classes)

print(result["top_class"])        # e.g. batik-megamendung
print(result["top_confidence"]) # e.g. 0.94
```

---

## Struktur Repositori

```text
Klasifikasi-Motif-Batik-Indonesia/
├── src/
│   ├── model.py            # Arsitektur CNN (ResNet18/50, SimpleCNN)
│   ├── dataset.py          # DataLoader & augmentasi
│   ├── train.py            # Script training
│   ├── inference.py        # Load model & prediksi
│   ├── utils.py            # Preprocessing & transform
│   └── merge_datasets.py   # Gabung multi-sumber dataset
├── app/
│   └── gradio_app.py       # Web interface
├── scripts/
│   ├── download_dataset.py # Unduh dataset dari URL
│   ├── prepare_dataset.py  # Split train/val
│   └── verify_dataset.py   # Validasi distribusi kelas
├── checkpoints/            # Model terlatih (via Git LFS)
│   ├── batik_best.pth
│   └── batik_resnet18_best.pth
├── data/
│   ├── raw/                # Dataset mentah (di-ignore, unduh manual)
│   └── processed/          # Dataset siap latih (di-ignore)
├── telegram_bot.py         # Bot Telegram
├── requirements.txt
└── Batik_Classifier_Notebook.ipynb
```

---

## Notebook Eksperimen

```bash
pip install jupyterlab
jupyter lab Batik_Classifier_Notebook.ipynb
```

---

## Lisensi Dataset

Setiap sumber dataset memiliki lisensi masing-masing. Pastikan mematuhi ketentuan lisensi saat menggunakan atau mendistribusikan data:

- **Roboflow Batik:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Kaggle datasets:** cek halaman dataset masing-masing
- Dataset lainnya: ikuti lisensi dari penyedia asli

---

*Dibuat untuk riset Klasifikasi Motif Batik Indonesia — Image Representation Learning.*