"""
TOPO-Transformer: Transformer with TOPO-2026 Prime-Anchored Continual Learning
=============================================================================

A production-ready implementation of TOPO-2026 for catastrophic forgetting
prevention in Transformer architectures.

Package Structure:
------------------
transformer-topo-poc/
├── src/               # Core source code
│   ├── topo_embedding.py    # Prime-anchored embedding layer
│   ├── topo_trainer.py      # Continual learning trainer
│   ├── topo_transformer.py  # Complete TOPO-Transformer
│   └── utils.py             # Utility functions
├── tests/             # Unit tests
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks
├── checkpoints/       # Saved model checkpoints
└── main.py           # Entry point

Quick Start:
------------
>>> from topo_transformer import TOPOTransformer, TOPOTransformerConfig
>>> config = TOPOTransformerConfig(vocab_size=1000, hidden_size=128)
>>> model = TOPOTransformer(config)
>>> # Train on Task 1
>>> result1 = trainer.train_task(loader1, epochs=5, task_name="task_1")
>>> # Train on Task 2 (won't forget Task 1)
>>> result2 = trainer.train_task(loader2, epochs=5, task_name="task_2")

Reference:
----------
Morales Aguilera, F. (2026). TOPO-2026: A Universal Solution to
Catastrophic Forgetting. Zenodo. DOI: 10.5281/zenodo.20785921
"""

__version__ = '1.0.0'
__author__ = 'Frank Morales Aguilera'
__email__ = 'frank.morales@sovereign-machine-lab.ai'
__license__ = 'MIT'

# Public API
from src import (
    TopologicalEmbedding,
    TOPOTrainer,
    TOPOTransformer,
    TOPOTransformerConfig,
    set_seed,
    get_device,
    count_parameters,
    create_synthetic_task,
    evaluate_model,
)

__all__ = [
    'TopologicalEmbedding',
    'TOPOTrainer',
    'TOPOTransformer',
    'TOPOTransformerConfig',
    'set_seed',
    'get_device',
    'count_parameters',
    'create_synthetic_task',
    'evaluate_model',
]
