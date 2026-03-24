# HSTU-MUSA: HSTU for Moore Threads (MUSA)

A MUSA-native implementation of [HSTU (Hierarchical Sequential Transduction Unit)](https://arxiv.org/abs/2402.17152)
ported from [NVIDIA recsys-examples](https://github.com/NVIDIA/recsys-examples),
with FBGEMM, TorchRec, and Megatron dependencies removed.

## Mathematical Equivalence

The model architecture is **mathematically equivalent** to the original recsys-examples HSTU.
The HSTU attention mask (contextual bidirectional attention, candidate position clamping,
target group masking, max attention length) is ported verbatim from the original
`pt_hstu_attention.py`. The only difference is the infrastructure layer:

| Component | Original (recsys-examples) | This Port |
|---|---|---|
| Embedding lookup | TorchRec EmbeddingCollection (FBGEMM TBE) | `torch.nn.Embedding` |
| Jagged tensors | `torchrec.sparse.jagged_tensor` | `compat/jagged_tensor.py` (pure PyTorch) |
| Jagged ops | FBGEMM GPU kernels | Pure PyTorch (`ops/jagged_ops.py`) |
| HSTU attention | CUTLASS / Triton / PyTorch | PyTorch (ported from `pt_hstu_attention.py`) |
| Linear layers | Megatron TEColumnParallelLinear | `torch.nn.Linear` |
| Distributed | Megatron DDP + TorchRec sharding | Single-GPU |

## Quick Start

```bash
pip install -r requirements.txt

# Train on MUSA with ML-20M
python train.py --device musa --backend pytorch

# Train on CPU (for debugging)
python train.py --device cpu --backend pytorch --nrows 500
```

## Accuracy Alignment with recsys-examples

Both hstu-musa and the original use the same hyperparameters from `movielen_ranking.gin`:

- batch_size=128, hidden_size=128, kv_channels=128, num_heads=4, num_layers=1
- prediction_head=[512, 10], dropout=0.2, bf16
- Adam(lr=1e-3, beta1=0.9, beta2=0.98), seed=1234
- ML-20M: 3 embeddings (user_id, movie_id, rating), max_candidates=20

To compare loss curves, run this on MUSA and the original on CUDA with the same
`processed_seqs.csv`, then compare `loss_hstu_musa.csv` step by step:

```bash
python train.py --loss-log loss_hstu_musa.csv --epochs 1
```

## Project Structure

```
hstu-musa/
├── compat/              # TorchRec compatibility shim
│   ├── jagged_tensor.py # JaggedTensor, KeyedJaggedTensor
│   └── embedding_config.py
├── ops/                 # Operators (pure PyTorch, no FBGEMM)
│   ├── jagged_ops.py    # Jagged tensor ops (cumsum, pad/unpad)
│   └── jagged_concat.py # Jagged tensor concatenation
├── modules/             # HSTU modules
│   ├── embedding.py     # torch.nn.Embedding based
│   ├── hstu_attention.py# Attention (ported from pt_hstu_attention.py)
│   ├── hstu_layer.py    # Single HSTU layer
│   ├── hstu_block.py    # Stack of HSTU layers
│   ├── hstu_processor.py# Pre/post processing
│   └── mlp.py           # MLP head
├── model/
│   └── ranking_gr.py    # Ranking model
├── configs/
│   └── hstu_config.py   # Configuration (matches movielen_ranking.gin)
├── data/
│   ├── batch.py         # HSTUBatch data structure
│   └── ml20m_dataset.py # ML-20M dataset loader
├── train.py             # Training script (MUSA-native)
└── requirements.txt
```
# HSTU-musa
