import os
import shutil
import zipfile
import requests
from io import BytesIO


def download_tiny_imagenet(data_dir="data/tiny-imagenet-200"):
    
    if os.path.exists(data_dir):
        print(f"Dataset already exists at {data_dir}, skipping download.")
        return
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print("Downloading Tiny-ImageNet...")
    response = requests.get(url)
    
    print("Extracting...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("data")
    
    # Reorganize validation set into class subdirectories
    # (required by ImageFolder)
    val_dir = os.path.join(data_dir, "val")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    
    print("Reorganizing validation set...")
    with open(annotations_file) as f:
        for line in f:
            fn, cls, *_ = line.split("\t")
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copyfile(
                os.path.join(val_dir, "images", fn),
                os.path.join(cls_dir, fn)
            )
    
    # Remove original images folder (now duplicates)
    shutil.rmtree(os.path.join(val_dir, "images"))
    
    print("Done! Dataset ready at", data_dir)


if __name__ == "__main__":
    download_tiny_imagenet()