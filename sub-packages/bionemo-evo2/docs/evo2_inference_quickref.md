# Evo2 Inference Flow - Quick Reference

## Simple Architecture Overview

```
                              ┌─────────────────────────────────┐
                              │         USER INPUTS             │
                              │  • FASTA file (sequences)       │
                              │  • Checkpoint directory         │
                              │  • Model configuration          │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │        DATA PIPELINE            │
                              │  SimpleFastaDataset             │
                              │  ┌────────────────────────────┐ │
                              │  │ 1. Load FASTA (NvFaidx)    │ │
                              │  │ 2. Tokenize (byte-level)   │ │
                              │  │ 3. Create position_ids     │ │
                              │  │ 4. Create loss_mask        │ │
                              │  │ 5. Optionally prepend BOS  │ │
                              │  └────────────────────────────┘ │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │      MODEL SELECTION            │
                              │  infer_model_type(model_size)   │
                              ├─────────┬──────────┬────────────┤
                              │ Hyena   │  Mamba   │   Llama    │
                              │(default)│ (hybrid) │  (Eden)    │
                              └────┬────┴────┬─────┴─────┬──────┘
                                   │         │           │
                              ┌────▼─────────▼───────────▼──────┐
                              │         PREDICTOR               │
                              │  BasePredictor + Model Wrapper  │
                              │  ┌────────────────────────────┐ │
                              │  │ • HyenaPredictor           │ │
                              │  │ • MambaPredictor           │ │
                              │  │ • LlamaPredictor           │ │
                              │  └────────────────────────────┘ │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │      MEGATRON STRATEGY          │
                              │  ┌────────────────────────────┐ │
                              │  │ • Tensor Parallel (TP)     │ │
                              │  │ • Context Parallel (CP)    │ │
                              │  │ • Pipeline Parallel (PP=1) │ │
                              │  │ • Sequence Parallel (SP)   │ │
                              │  └────────────────────────────┘ │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │       FORWARD PASS              │
                              │  hyena_predict_forward_step()   │
                              │  ┌────────────────────────────┐ │
                              │  │ Embedding → Decoder Stack  │ │
                              │  │ → Final Norm → LM Head     │ │
                              │  │                            │ │
                              │  │ Output: [B, S, V] logits   │ │
                              │  └────────────────────────────┘ │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │       POST-PROCESSING           │
                              │  BasePredictor.predict_step()   │
                              │  ┌────────────────────────────┐ │
                              │  │ 1. Gather across TP ranks  │ │
                              │  │ 2. Gather across CP ranks  │ │
                              │  │ 3. Compute log_probs (opt) │ │
                              │  │ 4. Apply loss_mask         │ │
                              │  └────────────────────────────┘ │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────▼──────────────────┐
                              │          OUTPUTS                │
                              │  PredictionWriter callback      │
                              │  ┌────────────────────────────┐ │
                              │  │ • token_logits [B, S, V]   │ │
                              │  │ • log_probs_seqs [B] or    │ │
                              │  │   [B, S] (per_token)       │ │
                              │  │ • pad_mask / loss_mask     │ │
                              │  │ • seq_idx_map.json         │ │
                              │  └────────────────────────────┘ │
                              └─────────────────────────────────┘
```

## Checkpoint Structure

```
checkpoint_dir/
├── context/
│   └── io.json                 # Serialization metadata
├── weights/
│   └── __0_0.distcp/           # Distributed checkpoint
│       ├── __0_0.pt            # Rank 0 shard
│       └── ...                 # Additional TP shards
├── model_config.yaml           # Architecture config
└── trainer_config.yaml         # Training config
```

## Model Architecture (Hyena/Evo2)

```
┌────────────────────────────────────────────────────────────────┐
│                     Evo2 Model (Hyena)                         │
├────────────────────────────────────────────────────────────────┤
│  Embedding Layer                                               │
│  └─ word_embeddings [vocab_size=512, hidden_size]             │
│  └─ input_layernorm                                           │
├────────────────────────────────────────────────────────────────┤
│  Decoder Stack (N layers)                                      │
│  │                                                             │
│  └─ Layer n:                                                   │
│     ├─ pre_mlp_layernorm                                       │
│     ├─ Hyena Mixer:                                           │
│     │  ├─ dense_projection (input proj)                       │
│     │  ├─ hyena_proj_conv (short conv)                        │
│     │  ├─ hyena_filter:                                       │
│     │  │  ├─ decay, gamma, h, p, R (learned filters)         │
│     │  │  └─ t (time embedding)                               │
│     │  ├─ short_conv (local convolution)                      │
│     │  ├─ rotary_emb (optional RoPE)                         │
│     │  └─ dense (output proj + bias)                          │
│     ├─ post_attention_layernorm                               │
│     ├─ outer_mlp_layernorm                                    │
│     └─ MLP:                                                   │
│        ├─ w1 (gate projection)                                │
│        ├─ w2 (up projection)                                  │
│        └─ w3 (down projection)                                │
├────────────────────────────────────────────────────────────────┤
│  Output Layer                                                  │
│  └─ final_layernorm (norm.weight)                             │
│  └─ output_layer (LM head, may tie with embeddings)          │
└────────────────────────────────────────────────────────────────┘
```

## Key Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fasta` | required | Input FASTA file |
| `--ckpt-dir` | required | Checkpoint directory |
| `--output-dir` | None | Where to save predictions |
| `--model-size` | 7b_arc_longcontext | Model configuration |
| `--tensor-parallel-size` | 1 | TP parallelism degree |
| `--context-parallel-size` | 1 | CP for long sequences |
| `--micro-batch-size` | 1 | Batch size per GPU |
| `--fp8` | False | Enable FP8 precision |
| `--full-fp8` | False | Full FP8 (vs vortex style) |
| `--output-log-prob-seqs` | False | Output log probs |
| `--log-prob-collapse-option` | mean | sum/mean/per_token |
| `--prepend-bos` | False | Add BOS token |
| `--lora-checkpoint-path` | None | LoRA weights path |

## Quick Start

```bash
# Basic inference
python -m bionemo.evo2.run.predict \
    --fasta input.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./output

# With parallelism for large models
python -m bionemo.evo2.run.predict \
    --fasta input.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./output \
    --tensor-parallel-size 4 \
    --context-parallel-size 2 \
    --devices 8

# Get sequence log probabilities
python -m bionemo.evo2.run.predict \
    --fasta input.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./output \
    --output-log-prob-seqs \
    --log-prob-collapse-option mean

```
---

## Context Parallelism for Long Sequences

### When to Use CP

Use Context Parallelism when your sequences are too long to fit in a single GPU's memory:

| Sequence Length | Recommendation |
|-----------------|----------------|
| ≤8K tokens      | CP=1 (no CP needed) |
| 8K-32K tokens   | CP=2 |
| 32K-128K tokens | CP=4 |
| 128K+ tokens    | CP=8+ |

### How CP Works (Zigzag Pattern)

CP uses **zigzag packing** to balance workload across GPUs:

```
Sequence: [0, 1, 2, 3, 4, 5, 6, 7]  (8 tokens, CP=2)

Split into 2*CP = 4 chunks:
  chunk_0: [0,1]  chunk_1: [2,3]  chunk_2: [4,5]  chunk_3: [6,7]

Zigzag assignment:
  GPU 0: chunk_0 + chunk_3 = [0, 1, 6, 7]
  GPU 1: chunk_1 + chunk_2 = [2, 3, 4, 5]

⚠️ Output is in ZIGZAG order, not original order!
```

### Code Flow

```python
# 1. Data step: Apply zigzag slicing
from megatron.core.utils import get_batch_on_this_cp_rank
batch = get_batch_on_this_cp_rank(batch)  # Slices for this CP rank

# 2. Forward pass: Each GPU processes its chunks
logits = model(batch["tokens"], batch["position_ids"])

# 3. Gather: Collect from all CP ranks
from bionemo.evo2.run.predict import _gather_along_cp_dim
full_logits = _gather_along_cp_dim(logits)  # Still in zigzag order!

# 4. (Optional) Unshuffle to original order
def unshuffle_zigzag(tensor, cp_size, seq_dim=1):
    chunks = tensor.chunk(2 * cp_size, dim=seq_dim)
    original = [None] * (2 * cp_size)
    for rank in range(cp_size):
        original[rank] = chunks[rank * 2]
        original[2*cp_size - rank - 1] = chunks[rank * 2 + 1]
    return torch.cat(original, dim=seq_dim)
```

### Requirements

- Sequence length must be divisible by `2 * CP_size`
- Use `--min-length` to pad shorter sequences
- PP=1 only (pipeline parallelism not supported with CP for inference)

### Example: Embedding 131K Token Sequences

```bash
python -m bionemo.evo2.run.predict \
    --fasta genome_sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --context-parallel-size 4 \
    --tensor-parallel-size 2 \
    --devices 8 \
    --micro-batch-size 1 \
    --min-length 131072  # Pad to ensure divisibility
```
