import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import numpy as np
from pathlib import Path
import json

class SeverityEstimatorModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        """
        Severity Estimator Model
        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value for severity)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SeverityDataset(Dataset):
    def __init__(self, embeddings_path: str, severity_labels_path: str):
        """
        Dataset for severity estimation
        Args:
            embeddings_path: Path to fused embeddings
            severity_labels_path: Path to severity labels
        """
        # Load embeddings
        data = torch.load(embeddings_path)
        if isinstance(data, torch.Tensor):
            self.embeddings = data
            self.sources = [f"sample_{i}" for i in range(len(data))]
        else:
            self.embeddings = data.get('embeddings', data)
            self.sources = data.get('sources', [f"sample_{i}" for i in range(len(self.embeddings))])
        
        # Load severity labels
        with open(severity_labels_path, 'r') as f:
            self.severity_labels = json.load(f)
        
        # Extract severity scores
        self.labels = []
        for source in self.sources:
            path = Path(source)
            # Try to get disease class from parent directory name
            disease_class = path.parent.name if isinstance(source, str) else "unknown"
            # Extract severity from filename or lookup in labels
            severity = self.severity_labels.get(disease_class, {}).get('severity', 0.5)
            self.labels.append(severity)
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

class SeverityEstimator:
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 learning_rate: float = 0.001,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Severity Estimator wrapper class
        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimization
            device: Device to use for computation
        """
        self.device = device
        self.model = SeverityEstimatorModel(input_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(embeddings).squeeze()
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                pred = self.model(embeddings).squeeze()
                loss = self.criterion(pred, labels)
                
                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'loss': total_loss / len(dataloader),
            'mae': np.mean(np.abs(predictions - actuals)),
            'rmse': np.sqrt(np.mean((predictions - actuals) ** 2))
        }
    
    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """Predict severity scores"""
        self.model.eval()
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            predictions = self.model(embeddings).squeeze()
            return predictions.cpu().numpy()
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 