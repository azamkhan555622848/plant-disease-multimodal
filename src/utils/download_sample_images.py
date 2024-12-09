import os
import requests
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile
import gdown

def download_file(url: str, destination: str, desc: str = "Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_from_gdrive(file_id: str, destination: str):
    """Download a file from Google Drive"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

def main():
    # Create directories if they don't exist
    base_dir = Path("data")
    images_dir = base_dir / "images_enhanced"
    os.makedirs(images_dir, exist_ok=True)
    
    # Google Drive file ID for ESRGAN-enhanced plant disease dataset
    file_id = "1mvjjqGQe4RRxV-Yd0PQedIKY9yHKJ4Ol"  # This is an example ID
    zip_path = base_dir / "esrgan_enhanced_images.zip"
    
    print("Downloading ESRGAN-enhanced plant disease images...")
    try:
        # First try Google Drive download
        download_from_gdrive(file_id, str(zip_path))
        
        print("\nExtracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print("\nDownload complete! ESRGAN-enhanced images are stored in data/images_enhanced/")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative download instructions:")
        print("1. Visit https://drive.google.com/file/d/1mvjjqGQe4RRxV-Yd0PQedIKY9yHKJ4Ol/view?usp=sharing")
        print("2. Download the ESRGAN-enhanced dataset")
        print("3. Extract the images to data/images_enhanced/")
        print("\nOr you can use regular plant disease images and enhance them with ESRGAN:")
        print("1. Visit https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("2. Download the dataset")
        print("3. Use ESRGAN to enhance the images")

if __name__ == "__main__":
    main() 