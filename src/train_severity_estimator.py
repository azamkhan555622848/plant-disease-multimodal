import torch
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path
import argparse
from models.severity_estimator import SeverityEstimator, SeverityDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def create_sample_severity_labels():
    """Create sample severity labels for different disease classes"""
    severity_labels = {
        "Apple___Apple_scab": {
            "severity": 0.7,
            "description": "High severity due to significant leaf damage"
        },
        "Apple___Black_rot": {
            "severity": 0.8,
            "description": "Very high severity with fruit rot and cankers"
        },
        "Corn___Common_rust": {
            "severity": 0.5,
            "description": "Moderate severity with pustules on leaves"
        },
        "Potato___Early_blight": {
            "severity": 0.6,
            "description": "Moderate to high severity with leaf lesions"
        },
        "Tomato___Leaf_Mold": {
            "severity": 0.4,
            "description": "Low to moderate severity with leaf discoloration"
        }
    }
    
    # Add more diseases with default severity
    for disease_class in ["Tomato___Late_blight", "Grape___Black_rot", "Orange___Haunglongbing"]:
        if disease_class not in severity_labels:
            severity_labels[disease_class] = {
                "severity": 0.5,
                "description": "Default moderate severity"
            }
    
    return severity_labels

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training History - MAE')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train(
    embeddings_path: str,
    severity_labels_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 50,
    val_split: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the severity estimator"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create severity labels
    if not Path(severity_labels_path).exists():
        print("Creating sample severity labels...")
        severity_labels = create_sample_severity_labels()
        with open(severity_labels_path, 'w') as f:
            json.dump(severity_labels, f, indent=2)
    
    # Create dataset
    dataset = SeverityDataset(embeddings_path, severity_labels_path)
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = dataset.embeddings.shape[1]
    model = SeverityEstimator(input_dim=input_dim, device=device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = model.train_epoch(train_loader)
        
        # Evaluate
        train_metrics = model.evaluate(train_loader)
        val_metrics = model.evaluate(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model.save(output_dir / 'best_model.pt')
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}")
        print("-" * 50)
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save(output_dir / 'final_model.pt')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Severity Estimator')
    parser.add_argument('--embeddings', type=str, required=True,
                      help='Path to fused embeddings')
    parser.add_argument('--labels', type=str, default='data/severity_labels.json',
                      help='Path to severity labels')
    parser.add_argument('--output', type=str, default='models/severity_estimator',
                      help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    train(
        embeddings_path=args.embeddings,
        severity_labels_path=args.labels,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

if __name__ == "__main__":
    main() 