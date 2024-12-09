import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List, Union, Tuple
import numpy as np

class CLIPEmbedder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def get_image_embeddings(self, images: List[Union[str, torch.Tensor]]) -> torch.Tensor:
        """
        Extract image embeddings using CLIP's image encoder
        Args:
            images: List of image paths or tensor images
        Returns:
            torch.Tensor: Image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        image_features = self.model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Extract text embeddings using CLIP's text encoder
        Args:
            texts: List of text descriptions
        Returns:
            torch.Tensor: Text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)
    
    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between image and text embeddings
        Args:
            image_embeddings: Tensor of image embeddings
            text_embeddings: Tensor of text embeddings
        Returns:
            torch.Tensor: Similarity scores
        """
        return torch.matmul(image_embeddings, text_embeddings.t())

    def fuse_embeddings(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                       method: str = "concatenate") -> torch.Tensor:
        """
        Fuse image and text embeddings using specified method
        Args:
            image_embeddings: Tensor of image embeddings
            text_embeddings: Tensor of text embeddings
            method: Fusion method ("concatenate", "add", "multiply")
        Returns:
            torch.Tensor: Fused embeddings
        """
        if method == "concatenate":
            return torch.cat([image_embeddings, text_embeddings], dim=-1)
        elif method == "add":
            return image_embeddings + text_embeddings
        elif method == "multiply":
            return image_embeddings * text_embeddings
        else:
            raise ValueError(f"Unknown fusion method: {method}")

class MultimodalClassifier(nn.Module):
    def __init__(self, clip_embedder: CLIPEmbedder, num_classes: int, fusion_method: str = "concatenate"):
        super().__init__()
        self.clip_embedder = clip_embedder
        self.fusion_method = fusion_method
        
        # Define input size based on fusion method
        if fusion_method == "concatenate":
            input_size = 512 * 2  # CLIP embeddings are 512-dimensional
        else:
            input_size = 512
            
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images: List[Union[str, torch.Tensor]], texts: List[str]) -> torch.Tensor:
        """
        Forward pass through the multimodal classifier
        Args:
            images: List of image paths or tensor images
            texts: List of text descriptions
        Returns:
            torch.Tensor: Classification logits
        """
        image_embeddings = self.clip_embedder.get_image_embeddings(images)
        text_embeddings = self.clip_embedder.get_text_embeddings(texts)
        fused_embeddings = self.clip_embedder.fuse_embeddings(
            image_embeddings, text_embeddings, self.fusion_method
        )
        return self.classifier(fused_embeddings) 