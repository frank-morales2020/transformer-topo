"""
TOPO-Transformer: Utility Functions
===================================

This module provides utility functions for the TOPO-Transformer package.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Any, Dict, List, Tuple
from pathlib import Path
import random
import logging
import json

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to: {seed}")


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
    **kwargs
):
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        **kwargs
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to load state into
        map_location: Device to load to
        
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info(f"Checkpoint loaded from: {filepath}")
    return checkpoint


def create_synthetic_task(
    task_id: int,
    num_samples: int = 500,
    vocab_size: int = 1000,
    seq_len: int = 20,
    num_classes: int = 2,
    seed: Optional[int] = None
) -> TensorDataset:
    """
    Create a synthetic classification task.
    
    Args:
        task_id: Unique task identifier
        num_samples: Number of samples
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        num_classes: Number of classes
        seed: Random seed (if None, uses task_id)
        
    Returns:
        TensorDataset with (input_ids, labels)
    """
    if seed is None:
        seed = task_id + 42
    torch.manual_seed(seed)
    
    # Random input tokens
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return TensorDataset(X, y)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader with evaluation data
        device: Device to use
        loss_fn: Optional loss function
        
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = batch.to(device)
            
            outputs = model(inputs)
            
            if loss_fn is not None:
                if isinstance(outputs, dict):
                    loss = loss_fn(outputs['logits'], inputs.get('labels'))
                else:
                    loss = loss_fn(outputs, inputs)
                total_loss += loss.item()
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            pred = logits.argmax(dim=-1)
            if isinstance(inputs, dict):
                labels = inputs.get('labels')
            else:
                labels = inputs
            
            if labels is not None:
                correct += (pred == labels).sum().item()
                total += labels.size(0)
    
    return {
        'loss': total_loss / len(loader) if loss_fn is not None else 0.0,
        'accuracy': correct / total if total > 0 else 0.0,
    }


def format_params(num_params: int) -> str:
    """Format parameter count in a human-readable way."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params
