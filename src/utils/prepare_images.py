import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

def prepare_images(input_dir: str, output_dir: str, max_images_per_class: int = 50):
    """
    Prepare images for CLIP processing:
    1. Organize images by class
    2. Convert to RGB if needed
    3. Resize if too large
    4. Copy to output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(input_path.rglob(f'*{ext}')))
        image_files.extend(list(input_path.rglob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get class name from parent directory
            class_name = img_path.parent.name
            
            # Create class directory in output
            class_output_dir = output_path / class_name
            class_output_dir.mkdir(exist_ok=True)
            
            # Count existing images in this class
            existing_images = len(list(class_output_dir.glob('*.*')))
            if existing_images >= max_images_per_class:
                continue
            
            # Open and process image
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (optional)
                if max(img.size) > 1024:
                    ratio = 1024 / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Save to output directory
                output_file = class_output_dir / f"{img_path.stem}_processed.jpg"
                img.save(output_file, 'JPEG', quality=95)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Prepare images for CLIP processing')
    parser.add_argument('--input_dir', type=str, default='data/images',
                      help='Input directory containing raw images')
    parser.add_argument('--output_dir', type=str, default='data/images_enhanced',
                      help='Output directory for processed images')
    parser.add_argument('--max_images', type=int, default=50,
                      help='Maximum number of images per class')
    
    args = parser.parse_args()
    
    prepare_images(args.input_dir, args.output_dir, args.max_images)
    print(f"\nImages prepared and saved to {args.output_dir}")

if __name__ == "__main__":
    main() 