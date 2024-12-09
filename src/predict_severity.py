import torch
import argparse
from pathlib import Path
import json
from models.severity_estimator import SeverityEstimator
import numpy as np
from PIL import Image
from models.clip_model import CLIPEmbedder

def load_model(model_path: str, input_dim: int, device: str) -> SeverityEstimator:
    """Load trained severity estimator model"""
    model = SeverityEstimator(input_dim=input_dim, device=device)
    model.load(model_path)
    return model

def predict_severity(
    model: SeverityEstimator,
    clip_model: CLIPEmbedder,
    image_path: str,
    text_description: str = None
) -> float:
    """
    Predict disease severity for a single image
    Args:
        model: Trained severity estimator
        clip_model: CLIP model for embedding extraction
        image_path: Path to input image
        text_description: Optional text description of the disease
    Returns:
        Predicted severity score (0-1)
    """
    # Get image embedding
    image = Image.open(image_path).convert('RGB')
    image_embedding = clip_model.get_image_embeddings([image])
    
    # Get text embedding if provided
    if text_description:
        text_embedding = clip_model.get_text_embeddings([text_description])
        # Fuse embeddings (using the same method as during training)
        embedding = torch.cat([image_embedding, text_embedding], dim=-1)
    else:
        embedding = image_embedding
    
    # Predict severity
    severity = model.predict(embedding)
    return float(severity.item() if isinstance(severity, torch.Tensor) else severity)

def main():
    parser = argparse.ArgumentParser(description='Predict Disease Severity')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained severity estimator model')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--text', type=str, default=None,
                      help='Text description of the disease (optional)')
    parser.add_argument('--input_dim', type=int, required=True,
                      help='Input dimension of the model')
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    model = load_model(args.model, args.input_dim, device)
    clip_model = CLIPEmbedder().to(device)
    
    # Predict severity
    print("Predicting severity...")
    severity = predict_severity(model, clip_model, args.image, args.text)
    
    # Print results
    print("\nResults:")
    print(f"Image: {args.image}")
    if args.text:
        print(f"Description: {args.text}")
    print(f"Predicted Severity: {severity:.2%}")
    
    # Interpret severity
    if severity < 0.3:
        level = "Low"
    elif severity < 0.6:
        level = "Moderate"
    else:
        level = "High"
    
    print(f"Severity Level: {level}")

if __name__ == "__main__":
    main() 