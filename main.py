#!/usr/bin/env python3
"""
TOPO-Transformer: Continual Learning POC

Run: python main.py --task train --epochs 10
"""

import argparse
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.topo_transformer import TOPOTransformer, TOPOTransformerConfig
from src.topo_trainer import TOPOTrainer
from src.topo_embedding import TopologicalEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_task(task_id, num_samples=500, vocab_size=1000, seq_len=20):
    """Create a synthetic classification task."""
    torch.manual_seed(task_id + 42)
    
    # Random input tokens
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Random labels (with some structure)
    y = torch.randint(0, 2, (num_samples,))
    
    return TensorDataset(X, y)

def main():
    parser = argparse.ArgumentParser(description="TOPO-Transformer POC")
    parser.add_argument("--task", type=str, default="train", choices=["train", "test", "demo"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model configuration
    config = TOPOTransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 2,
        num_labels=2,
    )
    
    # Initialize model
    model = TOPOTransformer(config)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Get embedding stats
    topo_emb = None
    for name, module in model.named_modules():
        if isinstance(module, TopologicalEmbedding):
            topo_emb = module
            logger.info(f"TopologicalEmbedding found: {topo_emb}")
            logger.info(f"  Anchor memory: {topo_emb.anchor_memory} bytes")
            logger.info(f"  Spectral coverage: {topo_emb.spectral_coverage:.4f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = TOPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_dir=save_dir,
        log_interval=10,
        eval_interval=100,
    )
    
    if args.task == "train":
        logger.info(f"Training on {args.num_tasks} tasks...")
        
        for task_id in range(1, args.num_tasks + 1):
            dataset = create_synthetic_task(
                task_id,
                num_samples=500,
                vocab_size=args.vocab_size,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            result = trainer.train_task(
                loader,
                epochs=args.epochs,
                task_name=f"task_{task_id}",
                take_snapshot=True,
            )
            
            logger.info(f"Task {task_id}: accuracy = {result['final_acc']:.4f}")
            
            # Verify anchors
            if topo_emb and topo_emb.verify_integrity():
                logger.info(f"✓ Anchors intact after task {task_id}")
            else:
                logger.warning(f"✗ Anchors corrupted after task {task_id}")
        
        logger.info("Training complete!")
        logger.info(f"Checkpoints saved to: {save_dir}")
        
        # Final stats
        logger.info(f"Total tasks learned: {trainer._task_id}")
        logger.info(f"Anchor memory: {topo_emb.anchor_memory if topo_emb else 0} bytes")
    
    elif args.task == "test":
        logger.info("Running tests...")
        pytest.main(["tests/"])
    
    elif args.task == "demo":
        logger.info("Running demo...")
        # Quick demo
        dataset1 = create_synthetic_task(1, num_samples=100)
        loader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
        
        result = trainer.train_task(loader1, epochs=2, task_name="demo")
        logger.info(f"Demo completed: accuracy = {result['final_acc']:.4f}")

if __name__ == "__main__":
    main()
