import shutil
from pathlib import Path

# Mapping dari dataset baru ke kelas kita
class_mapping = {
    # Dataset 1
    'Gentongan': 'batik-gentongan',
    'IkatCelup': 'batik-celup',
    'Kawung': 'batik-kawung',
    'Megamendung': 'batik-megamendung',
    'Parang': 'batik-parang',
    # Dataset 2
    'sidomukti': 'batik-sidomukti'
}

data_raw = Path('/root/batik-cnn-classifier/data/raw')
data_processed = Path('/root/batik-cnn-classifier/data/processed')

# Source paths
dataset1_train = data_raw / 'dataset1_extracted' / 'train'
dataset2_train = data_raw / 'dataset2_extracted' / 'train'

# Backup current processed
backup_path = data_raw / 'processed_backup_before_merge'
if data_processed.exists():
    print(f"Backing up current dataset to {backup_path}...")
    shutil.copytree(data_processed, backup_path, dirs_exist_ok=True)
    print("Backup complete!")

# Merge dataset 1
print("\n=== Merging Dataset 1 ===")
for src_class, dst_class in class_mapping.items():
    if src_class == 'sidomukti':
        continue  # Skip, will handle from dataset2
    
    src_path = dataset1_train / src_class
    dst_train_path = data_processed / 'train' / dst_class
    dst_val_path = data_processed / 'val' / dst_class
    
    if not src_path.exists():
        print(f"  Skipping {src_class} (not found)")
        continue
    
    # Get all images
    images = list(src_path.glob('*.jpg')) + list(src_path.glob('*.jpeg')) + list(src_path.glob('*.png'))
    
    # Split 80/20
    import random
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy to train
    dst_train_path.mkdir(parents=True, exist_ok=True)
    for img in train_images:
        dst_img = dst_train_path / f"ds1_{img.name}"
        shutil.copy2(img, dst_img)
    
    # Copy to val
    dst_val_path.mkdir(parents=True, exist_ok=True)
    for img in val_images:
        dst_img = dst_val_path / f"ds1_{img.name}"
        shutil.copy2(img, dst_img)
    
    print(f"  {src_class} → {dst_class}: +{len(train_images)} train, +{len(val_images)} val")

# Merge dataset 2 (sidomukti)
print("\n=== Merging Dataset 2 (Sidomukti) ===")
src_path = dataset2_train / 'sidomukti'
dst_class = 'batik-sidomukti'
dst_train_path = data_processed / 'train' / dst_class
dst_val_path = data_processed / 'val' / dst_class

images = list(src_path.glob('*.jpg')) + list(src_path.glob('*.jpeg')) + list(src_path.glob('*.png'))

import random
random.seed(42)
random.shuffle(images)
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Copy to train
dst_train_path.mkdir(parents=True, exist_ok=True)
for img in train_images:
    dst_img = dst_train_path / f"ds2_{img.name}"
    shutil.copy2(img, dst_img)

# Copy to val
dst_val_path.mkdir(parents=True, exist_ok=True)
for img in val_images:
    dst_img = dst_val_path / f"ds2_{img.name}"
    shutil.copy2(img, dst_img)

print(f"  sidomukti → {dst_class}: +{len(train_images)} train, +{len(val_images)} val")

print("\n=== Final Dataset Statistics ===")
for split in ['train', 'val']:
    split_path = data_processed / split
    print(f"\n{split.upper()} set:")
    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.jpeg'))) + len(list(class_dir.glob('*.png')))
            print(f"  {class_dir.name}: {count}")

print("\n✅ Merge complete!")
