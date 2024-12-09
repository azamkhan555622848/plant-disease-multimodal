import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Union
import os
from models.clip_model import CLIPEmbedder
from PIL import Image

# Define the paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images_enhanced" / "New Plant Diseases Dataset(Augmented)" / "train"
AGROLD_DIR = DATA_DIR / "agrold"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

def load_image(image_path: str) -> Image.Image:
    """Load a single image file"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img.copy()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_images(input_dir: str) -> List[str]:
    """Load image paths from directory"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    print(f"Searching for images in: {input_dir.absolute()}")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
    if not image_paths:
        raise ValueError(
            f"No valid images found in {input_dir}. "
            f"Supported formats: {', '.join(image_extensions)}"
        )
    
    print(f"Total images found: {len(image_paths)}")
    return sorted(image_paths)

def process_batch(paths: List[str], mode: str = "image") -> List[Union[Image.Image, str]]:
    """Process a batch of data"""
    if mode == "image":
        # Load images and filter out None values
        batch = [load_image(path) for path in paths]
        return [img for img in batch if img is not None]
    return paths  # For text mode, return as is

def load_text_data(input_file: str) -> List[str]:
    """Load text data from file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_embeddings(
    mode: str,
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Extract CLIP embeddings from images or text
    Args:
        mode: Either 'image' or 'text'
        input_path: Path to input directory (images) or file (text)
        output_path: Path to save embeddings
        batch_size: Batch size for processing
        device: Device to use for computation
    """
    print(f"Using device: {device}")
    print(f"Processing {mode} data from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize CLIP model
    clip_embedder = CLIPEmbedder().to(device)
    clip_embedder.eval()

    # Load data based on mode
    if mode == "image":
        data_paths = load_images(input_path)
        embedding_function = clip_embedder.get_image_embeddings
    else:  # text mode
        data_paths = load_text_data(input_path)
        embedding_function = clip_embedder.get_text_embeddings

    # Process in batches
    all_embeddings = []
    valid_paths = []  # Keep track of successfully processed paths
    
    for i in tqdm(range(0, len(data_paths), batch_size), desc=f"Processing {mode} data"):
        batch_paths = data_paths[i:i + batch_size]
        batch_data = process_batch(batch_paths, mode)
        
        if not batch_data:  # Skip empty batches
            continue
            
        with torch.no_grad():
            try:
                embeddings = embedding_function(batch_data)
                all_embeddings.append(embeddings.cpu())
                valid_paths.extend(batch_paths[:len(batch_data)])  # Only include paths for successful embeddings
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    if not all_embeddings:
        raise ValueError("No valid embeddings were generated")

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings and paths/texts
    torch.save({
        'embeddings': all_embeddings,
        'sources': valid_paths
    }, output_path)
    
    print(f"Saved {len(valid_paths)} embeddings to {output_path}")

def main():
    # Create necessary directories
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Process images
    print("\nProcessing Images...")
    extract_embeddings(
        mode="image",
        input_path=str(IMAGES_DIR),
        output_path=str(EMBEDDINGS_DIR / "images_embeddings.pt"),
        batch_size=32
    )
    
    # Process text
    print("\nProcessing Text...")
    extract_embeddings(
        mode="text",
        input_path=str(AGROLD_DIR / "descriptions.txt"),
        output_path=str(EMBEDDINGS_DIR / "text_embeddings.pt"),
        batch_size=32
    )
    
    print("\nAll embeddings have been extracted successfully!")

if __name__ == "__main__":
    main() 