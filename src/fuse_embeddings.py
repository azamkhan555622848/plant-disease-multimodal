import argparse
import torch
import os
from typing import Dict, Tuple
from models.clip_model import CLIPEmbedder

def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load embeddings from file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embeddings found at {path}")
    return torch.load(path)

def align_embeddings(
    image_data: Dict[str, torch.Tensor],
    text_data: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align image and text embeddings based on their sources
    Returns aligned embeddings
    """
    # This is a simple implementation - you might need to modify this
    # based on your specific naming conventions and matching requirements
    image_embeddings = image_data['embeddings']
    text_embeddings = text_data['embeddings']
    
    # Ensure we have the same number of embeddings
    min_len = min(len(image_embeddings), len(text_embeddings))
    return image_embeddings[:min_len], text_embeddings[:min_len]

def fuse_embeddings(
    image_embeddings_path: str,
    text_embeddings_path: str,
    output_path: str,
    fusion_method: str = "concatenate",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Load and fuse image and text embeddings
    Args:
        image_embeddings_path: Path to image embeddings
        text_embeddings_path: Path to text embeddings
        output_path: Path to save fused embeddings
        fusion_method: Method to use for fusion
        device: Device to use for computation
    """
    # Load embeddings
    image_data = load_embeddings(image_embeddings_path)
    text_data = load_embeddings(text_embeddings_path)
    
    # Align embeddings
    image_embeddings, text_embeddings = align_embeddings(image_data, text_data)
    
    # Initialize CLIP embedder for fusion
    clip_embedder = CLIPEmbedder().to(device)
    
    # Move embeddings to device
    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)
    
    # Fuse embeddings
    fused_embeddings = clip_embedder.fuse_embeddings(
        image_embeddings,
        text_embeddings,
        method=fusion_method
    )
    
    # Save fused embeddings
    torch.save({
        'embeddings': fused_embeddings.cpu(),
        'image_sources': image_data['sources'][:len(fused_embeddings)],
        'text_sources': text_data['sources'][:len(fused_embeddings)],
        'fusion_method': fusion_method
    }, output_path)
    
    print(f"Saved {len(fused_embeddings)} fused embeddings to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fuse image and text embeddings')
    parser.add_argument('--image_embeddings', type=str, required=True,
                      help='Path to image embeddings')
    parser.add_argument('--text_embeddings', type=str, required=True,
                      help='Path to text embeddings')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save fused embeddings')
    parser.add_argument('--fusion_method', type=str, default="concatenate",
                      choices=["concatenate", "add", "multiply"],
                      help='Method to use for fusion')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    fuse_embeddings(
        image_embeddings_path=args.image_embeddings,
        text_embeddings_path=args.text_embeddings,
        output_path=args.output,
        fusion_method=args.fusion_method
    )

if __name__ == "__main__":
    main() 