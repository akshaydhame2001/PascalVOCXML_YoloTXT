import os
import random
import shutil

# Paths
dataset_path = "dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Output folders
output_dirs = {
    "train": {"images": "dataset_split/train/images", "labels": "dataset_split/train/labels"},
    "val": {"images": "dataset_split/val/images", "labels": "dataset_split/val/labels"},
    "test": {"images": "dataset_split/test/images", "labels": "dataset_split/test/labels"},
}

# Create output directories
for split, paths in output_dirs.items():
    for folder in paths.values():
        os.makedirs(folder, exist_ok=True)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# List all images
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle and split
random.seed(42)  # For reproducibility
random.shuffle(image_files)
total_files = len(image_files)

train_split = int(total_files * train_ratio)
val_split = int(total_files * (train_ratio + val_ratio))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# Function to move files
def move_files(file_list, split):
    for file in file_list:
        image_src = os.path.join(images_path, file)
        label_src = os.path.join(labels_path, file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Destination paths
        image_dst = os.path.join(output_dirs[split]["images"], file)
        label_dst = os.path.join(output_dirs[split]["labels"], os.path.basename(label_src))
        
        # Move image and label
        shutil.copy(image_src, image_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

# Move files to respective folders
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"Data split completed:")
print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
