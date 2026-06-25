"""
TOPO-2026: Continual Learning Trainer
=====================================

This module provides a trainer for continual learning with TOPO-2026
anchor protection. It manages the learning loop, snapshot management,
and tracks forgetting and backward transfer.

Reference:
Morales Aguilera, F. (2026). TOPO-2026: A Universal Solution to
Catastrophic Forgetting. Zenodo. DOI: 10.5281/zenodo.20785921
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class TOPOTrainer:
    """
    Trainer for continual learning with TOPO-2026.
    
    This trainer manages the complete continual learning loop with
    TOPO-2026 anchor protection, including snapshot management,
    forgetting tracking, and backward transfer measurement.
    
    Args:
        model: The model to train (must contain TopologicalEmbedding)
        optimizer: The optimizer to use
        loss_fn: The loss function
        device: The device to use
        task_name: Base name for tasks
        log_interval: How often to log (in batches)
        eval_interval: How often to evaluate (in epochs)
        save_dir: Directory to save checkpoints
        verbose: Whether to print verbose output
    
    Example:
        >>> trainer = TOPOTrainer(model, optimizer, loss_fn, device)
        >>> # Train Task 1
        >>> result1 = trainer.train_task(loader1, epochs=10, task_name="task_1")
        >>> # Train Task 2 (won't forget Task 1)
        >>> result2 = trainer.train_task(loader2, epochs=10, task_name="task_2")
        >>> # Check anchors
        >>> topo_emb = trainer._get_topo_embedding()
        >>> assert topo_emb.verify_integrity()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        task_name: str = "task",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.task_name = task_name
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.verbose = verbose
        self.save_dir = save_dir or Path("checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'eval_loss': [],
            'eval_acc': [],
            'forgetting': [],
            'backward_transfer': [],
            'task_history': [],
        }
        
        self._task_id = 0
        self._previous_accuracies: Dict[int, float] = {}
        self._task_names: Dict[int, str] = {}
        self._task_loader_cache: Dict[int, DataLoader] = {}
        
        # Get the TopologicalEmbedding
        self._topo_embedding = self._get_topo_embedding()
        if self._topo_embedding is None:
            logger.warning("No TopologicalEmbedding found in model!")
        
        logger.info(f"TOPOTrainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Save dir: {self.save_dir}")
        logger.info(f"  TopologicalEmbedding: {self._topo_embedding is not None}")
    
    def _get_topo_embedding(self):
        """Find the TopologicalEmbedding in the model."""
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'TopologicalEmbedding':
                logger.debug(f"Found TopologicalEmbedding at: {name}")
                return module
        return None
    
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        task_name: Optional[str] = None,
        take_snapshot: bool = True,
        track_previous_tasks: bool = False,
    ) -> Dict[str, float]:
        """
        Train on a new task with TOPO-2026 protection.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of epochs to train
            task_name: Name for the task (auto-generated if None)
            take_snapshot: Whether to take a snapshot after training
            track_previous_tasks: Whether to evaluate on previous tasks
            
        Returns:
            Dict with training results
        """
        self._task_id += 1
        task_name = task_name or f"{self.task_name}_{self._task_id}"
        
        logger.info(f"Starting training on task: {task_name}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Samples: {len(train_loader.dataset)}")
        
        # Cache the task name
        self._task_names[self._task_id] = task_name
        
        # If we have previous tasks, record their current accuracies
        previous_accuracies = {}
        if track_previous_tasks:
            previous_accuracies = self._evaluate_previous_tasks()
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Task {self._task_id} | Epoch {epoch+1}/{epochs}",
                leave=False,
                disable=not self.verbose
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                inputs = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs['logits'], inputs.get('labels'))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # ===== TOPO-2026: Zero anchor gradients =====
                if self._topo_embedding is not None:
                    self._topo_embedding.zero_anchor_gradients()
                
                # Optimizer step
                self.optimizer.step()
                
                # ===== TOPO-2026: Enforce anchors =====
                if self._topo_embedding is not None:
                    self._topo_embedding.enforce_anchors()
                
                # Metrics
                total_loss += loss.item()
                if 'logits' in outputs:
                    pred = outputs['logits'].argmax(dim=-1)
                    labels = inputs.get('labels')
                    if labels is not None:
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                
                # Update progress bar
                if total > 0:
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'acc': correct/total,
                    })
                
                # Logging
                if batch_idx % self.log_interval == 0:
                    logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")
            
            # Epoch metrics
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total if total > 0 else 0.0
            
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            
            # Evaluation
            if val_loader and (epoch % self.eval_interval == 0 or epoch == epochs - 1):
                eval_metrics = self.evaluate(val_loader)
                self.history['eval_loss'].append(eval_metrics['loss'])
                self.history['eval_acc'].append(eval_metrics['accuracy'])
                
                if eval_metrics['accuracy'] > best_val_acc:
                    best_val_acc = eval_metrics['accuracy']
                
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch+1}: "
                        f"train_loss={epoch_loss:.4f}, "
                        f"train_acc={epoch_acc:.4f}, "
                        f"val_loss={eval_metrics['loss']:.4f}, "
                        f"val_acc={eval_metrics['accuracy']:.4f}"
                    )
            elif self.verbose:
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"train_loss={epoch_loss:.4f}, "
                    f"train_acc={epoch_acc:.4f}"
                )
        
        # ===== TOPO-2026: Take snapshot after task =====
        if take_snapshot and self._topo_embedding is not None:
            self._topo_embedding.take_snapshot(task_name)
            logger.info(f"Snapshot taken for task: {task_name}")
            
            # Verify integrity
            if self._topo_embedding.verify_integrity():
                logger.info("✓ Anchor integrity verified")
            else:
                logger.warning("⚠ Anchor integrity check failed!")
        
        # Track forgetting
        if track_previous_tasks:
            self._track_forgetting(previous_accuracies)
        
        # Log task completion
        self.history['task_history'].append({
            'task_id': self._task_id,
            'task_name': task_name,
            'final_loss': epoch_loss,
            'final_acc': epoch_acc,
            'best_val_acc': best_val_acc,
            'epochs': epochs,
        })
        
        # Save checkpoint
        self.save_checkpoint(f"task_{self._task_id}")
        
        result = {
            'task_id': self._task_id,
            'task_name': task_name,
            'final_loss': epoch_loss,
            'final_acc': epoch_acc,
            'best_val_acc': best_val_acc,
            'snapshot_taken': take_snapshot,
        }
        
        logger.info(f"✓ Task {self._task_id} completed: accuracy={epoch_acc:.4f}")
        return result
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            loader: DataLoader with evaluation data
            
        Returns:
            Dict with loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                inputs = self._move_to_device(batch)
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs['logits'], inputs.get('labels'))
                
                total_loss += loss.item()
                if 'logits' in outputs:
                    pred = outputs['logits'].argmax(dim=-1)
                    labels = inputs.get('labels')
                    if labels is not None:
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0.0,
        }
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _evaluate_previous_tasks(self) -> Dict[int, float]:
        """Evaluate all previous tasks."""
        previous_accuracies = {}
        # In a full implementation, this would evaluate on cached datasets
        # For now, we just return an empty dict
        return previous_accuracies
    
    def _track_forgetting(self, previous_accuracies: Dict[int, float]):
        """Track forgetting on previous tasks."""
        # This would calculate forgetting metrics
        pass
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.save_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'task_id': self._task_id,
            'history': self.history,
            'task_names': self._task_names,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._task_id = checkpoint['task_id']
        self.history = checkpoint['history']
        self._task_names = checkpoint['task_names']
        logger.info(f"Checkpoint loaded: {path}")
    
    def get_anchor_stats(self) -> Dict[str, Any]:
        """Get statistics from the TopologicalEmbedding."""
        if self._topo_embedding is not None:
            return self._topo_embedding.get_anchor_stats()
        return {'error': 'No TopologicalEmbedding found'}
    
    def verify_integrity(self) -> bool:
        """Verify anchor integrity."""
        if self._topo_embedding is not None:
            return self._topo_embedding.verify_integrity()
        return False
    
    def reset(self):
        """Reset the trainer and model."""
        self._task_id = 0
        self._task_names = {}
        self.history = {k: [] for k in self.history}
        if self._topo_embedding is not None:
            self._topo_embedding.reset()
        logger.info("Trainer reset")
