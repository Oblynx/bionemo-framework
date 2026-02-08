# Evo2 Inference Architecture

## Overview

This document provides a comprehensive architectural overview of the Evo2 inference system in BioNeMo Framework. Evo2 is a genomic foundation model that supports multiple backbone architectures (Hyena, Mamba, and Llama) and is designed for DNA/RNA sequence analysis and generation.

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              EVO2 INFERENCE PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────────┐
                                    │   User Input     │
                                    │  (FASTA File)    │
                                    └────────┬─────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA LAYER                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         SimpleFastaDataset                                        │   │
│  │  ┌─────────────┐    ┌───────────────┐    ┌─────────────────┐                     │   │
│  │  │   NvFaidx   │───▶│   Tokenizer   │───▶│  Batch Creation │                     │   │
│  │  │ (FASTA I/O) │    │ (byte-level)  │    │  (tokens, pos,  │                     │   │
│  │  └─────────────┘    └───────────────┘    │   loss_mask)    │                     │   │
│  │                                          └─────────────────┘                     │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                             │                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         PredictDataModule                                         │   │
│  │  • Wraps dataset with WrappedDataLoader                                          │   │
│  │  • Handles padding via padding_collate_fn                                        │   │
│  │  • Supports variable sequence lengths                                            │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEGATRON DATA STEP                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                      hyena_predict_data_step()                                    │   │
│  │  • Extracts batch from dataloader iterator                                       │   │
│  │  • Handles pipeline parallelism staging                                          │   │
│  │  • Moves tensors to GPU (non-blocking)                                           │   │
│  │  • Applies Context Parallelism (CP) slicing via get_batch_on_this_cp_rank()     │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL LAYER                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Model Selection (by model_size)                            │   │
│  │                                                                                   │   │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │   │
│  │   │  HYENA_MODEL_    │  │  MAMBA_MODEL_    │  │  LLAMA_MODEL_    │              │   │
│  │   │    OPTIONS       │  │    OPTIONS       │  │    OPTIONS       │              │   │
│  │   │ (from NeMo)      │  │ (hybrid_mamba_8b)│  │ (Eden configs)   │              │   │
│  │   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │   │
│  │            │                      │                      │                       │   │
│  │            ▼                      ▼                      ▼                       │   │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │   │
│  │   │  HyenaPredictor  │  │  MambaPredictor  │  │  LlamaPredictor  │              │   │
│  │   │  (HyenaModel)    │  │  (MambaModel)    │  │  (GPTModel)      │              │   │
│  │   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │   │
│  │            │                      │                      │                       │   │
│  │            └──────────────────────┼──────────────────────┘                       │   │
│  │                                   │                                              │   │
│  │                                   ▼                                              │   │
│  │                    ┌──────────────────────────────┐                              │   │
│  │                    │     BasePredictor            │                              │   │
│  │                    │ (LightningPassthrough        │                              │   │
│  │                    │  PredictionMixin)            │                              │   │
│  │                    └──────────────────────────────┘                              │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FORWARD PASS                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                     hyena_predict_forward_step()                                  │   │
│  │  • Input: {input_ids, position_ids, attention_mask}                              │   │
│  │  • Handles packed sequence parameters (if cu_seqlens present)                    │   │
│  │  • Returns: logits tensor [batch, seq_len, vocab_size]                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                             │                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Model Architecture (Hyena Example)                         │   │
│  │                                                                                   │   │
│  │    ┌─────────────────┐                                                           │   │
│  │    │   Embedding     │  word_embeddings.weight [vocab_size, hidden_size]         │   │
│  │    │   + LayerNorm   │  input_layernorm.weight                                   │   │
│  │    └────────┬────────┘                                                           │   │
│  │             │                                                                     │   │
│  │             ▼                                                                     │   │
│  │    ┌─────────────────────────────────────────────────────────────────────┐       │   │
│  │    │                    Decoder Stack (N layers)                         │       │   │
│  │    │  ┌─────────────────────────────────────────────────────────────┐   │       │   │
│  │    │  │  Layer N:                                                    │   │       │   │
│  │    │  │  ├── pre_mlp_layernorm                                      │   │       │   │
│  │    │  │  ├── mixer (Hyena/Mamba/Attention)                          │   │       │   │
│  │    │  │  │   ├── dense_projection (linear)                          │   │       │   │
│  │    │  │  │   ├── hyena_proj_conv (short conv filter)                │   │       │   │
│  │    │  │  │   ├── hyena_filter (decay, gamma, h, p, R)              │   │       │   │
│  │    │  │  │   ├── short_conv                                         │   │       │   │
│  │    │  │  │   ├── rotary_emb (optional RoPE)                        │   │       │   │
│  │    │  │  │   └── dense (output projection)                          │   │       │   │
│  │    │  │  ├── post_attention_layernorm                               │   │       │   │
│  │    │  │  ├── outer_mlp_layernorm                                    │   │       │   │
│  │    │  │  └── mlp                                                    │   │       │   │
│  │    │  │      ├── w1 (gate)                                          │   │       │   │
│  │    │  │      ├── w2 (up projection)                                 │   │       │   │
│  │    │  │      └── w3 (down projection)                               │   │       │   │
│  │    │  └─────────────────────────────────────────────────────────────┘   │       │   │
│  │    └─────────────────────────────────────────────────────────────────────┘       │   │
│  │             │                                                                     │   │
│  │             ▼                                                                     │   │
│  │    ┌─────────────────┐                                                           │   │
│  │    │   Final Norm    │  norm.weight                                              │   │
│  │    │   + Output LM   │  (tied with embeddings or separate)                       │   │
│  │    │     Head        │                                                           │   │
│  │    └─────────────────┘                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           PREDICT STEP (Post-Processing)                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         BasePredictor.predict_step()                              │   │
│  │                                                                                   │   │
│  │   1. Forward pass → forward_out (logits)                                         │   │
│  │   2. Gather across Tensor Parallelism: _gather_along_last_dim()                  │   │
│  │   3. Gather across Context Parallelism: _gather_along_cp_dim()                   │   │
│  │                                                                                   │   │
│  │   Output Options:                                                                 │   │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│  │   │  output_log_prob_seqs=True          │  output_log_prob_seqs=False      │   │   │
│  │   │  ─────────────────────────────────  │  ──────────────────────────────  │   │   │
│  │   │  • Compute log_softmax              │  • Return raw token_logits       │   │   │
│  │   │  • Gather log probs at token IDs    │  • Return pad_mask               │   │   │
│  │   │  • Apply loss_mask                  │  • Return seq_idx                │   │   │
│  │   │  • Collapse: sum/mean/per_token     │                                  │   │   │
│  │   └─────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          PredictionWriter Callback                                │   │
│  │  • Saves predictions to output_dir                                               │   │
│  │  • Supports batch or epoch write intervals                                       │   │
│  │  • Creates seq_idx_map.json for sequence mapping                                 │   │
│  │  • Handles subdirectory organization (files_per_subdir)                          │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Checkpoint Structure

### NeMo2 Checkpoint Format

Evo2 checkpoints follow the NeMo2 distributed checkpoint format. The primary format is `torch_dist` (distributed PyTorch), though `zarr` is also supported (deprecated).

```
checkpoint_dir/
├── context/
│   └── io.json                    # Serialization context
├── weights/
│   ├── __0_0.distcp/              # Distributed checkpoint shards
│   │   ├── __0_0.pt               # Weight shard for rank 0
│   │   ├── __0_1.pt               # Weight shard for rank 1 (if TP>1)
│   │   └── ...
│   └── metadata.json              # Checkpoint metadata
├── model_config.yaml              # Model architecture configuration
└── trainer_config.yaml            # Training configuration
```

### Key Checkpoint Parameters (Hyena/Evo2)

Based on `EVO2_PARAMS` in [params.py](src/bionemo/evo2/utils/checkpoint/params.py):

| Parameter | Shape | Partition Dim | Description |
|-----------|-------|---------------|-------------|
| `word_embeddings.weight` | [vocab, hidden] | 0 | Token embeddings |
| `input_layernorm.weight` | [hidden] | None | Input layer normalization |
| `mixer.dense_projection.weight` | [proj, hidden] | 0 | Hyena projection |
| `mixer.hyena_proj_conv.short_conv_weight` | [proj, 3] | 0 | Short convolution |
| `mixer.mixer.filter.decay` | [heads, seq] | 0 | Hyena decay filter |
| `mixer.mixer.filter.gamma` | [heads, dim] | 0 | Hyena gamma |
| `mixer.mixer.filter.h` | [heads, seq] | 0 | Hyena H matrix |
| `mixer.mixer.filter.p` | [heads, dim] | 0 | Hyena P matrix |
| `mixer.mixer.filter.R` | [heads, dim] | 0 | Hyena R matrix |
| `mixer.mixer.filter.t` | [1, 1, seq] | None | Time embedding |
| `mixer.dense.weight` | [hidden, proj] | 1 | Output projection |
| `mlp.w1.weight` | [ffn, hidden] | 0 | Gate projection |
| `mlp.w2.weight` | [ffn, hidden] | 0 | Up projection |
| `mlp.w3.weight` | [hidden, ffn] | 1 | Down projection |
| `norm.weight` | [hidden] | None | Final layer norm |

---

## Component Details

### 1. Entry Points

There are two main inference modes:

#### A. Forward Pass Prediction (`predict.py`)
- **Purpose**: Compute logits or log probabilities for sequences
- **Use Case**: Scoring sequences, computing perplexity, embeddings extraction
- **Entry**: `python -m bionemo.evo2.run.predict --fasta input.fa --ckpt-dir /path/to/ckpt`

#### B. Autoregressive Generation (`infer.py`)
- **Purpose**: Generate new sequences from a prompt
- **Use Case**: DNA/RNA sequence generation
- **Entry**: `python -m bionemo.evo2.run.infer --prompt "|d__Bacteria;..." --ckpt-dir /path/to/ckpt`

### 2. Model Configurations

#### Hyena Models (Default Evo2)
Configured via NeMo's `HYENA_MODEL_OPTIONS`:
- `7b_arc_longcontext` - 7B parameter long-context Hyena model
- Custom hybrid patterns via `hybrid_override_pattern`

#### Mamba Models
Configured via `MAMBA_MODEL_OPTIONS`:
- `hybrid_mamba_8b` - 8B hybrid Mamba with attention layers

Key config parameters:
```python
@dataclass
class HybridMambaConfig8BEvo2Loss(NemotronHConfigBase):
    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-..."  # Layer pattern
    num_layers: int = 52
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    ffn_hidden_size: int = 21504
    vocab_size: int = 512  # Byte-level tokenizer
```

#### Llama/Eden Models
Configured via `LLAMA_MODEL_OPTIONS`:
- Multiple Eden configs (8B, 11B, 18B, 21B, 24B, 27B, 28B, 35B)
- Based on Llama 3.1 architecture with RoPE

### 3. Parallelism Support

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM DIMENSIONS                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tensor Parallelism (TP)                                    │   │
│  │  • Shards weight matrices across GPUs                       │   │
│  │  • sequence_parallel enabled when TP > 1                    │   │
│  │  • Gathers outputs via _gather_along_last_dim()            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Context Parallelism (CP)                                   │   │
│  │  • Splits sequences across GPUs for long contexts           │   │
│  │  • Uses zigzag packing via get_batch_on_this_cp_rank()     │   │
│  │  • Gathers via _gather_along_cp_dim()                      │   │
│  │  • Note: per_token log probs shuffled with CP > 1          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Pipeline Parallelism (PP)                                  │   │
│  │  • Currently only PP=1 is supported for inference           │   │
│  │  • First/last stage handling in data_step                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  world_size = num_nodes × devices                                   │
│  model_parallel_size = TP × PP × CP                                │
│  global_batch_size = micro_batch_size × (world_size / model_parallel_size)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Context Parallelism Deep Dive

Context Parallelism (CP) is essential for embedding **long sequences** that don't fit in a single GPU's memory. This section provides a detailed explanation of how CP works in Evo2 inference.

### What is Context Parallelism?

Context Parallelism distributes a single sequence across multiple GPUs. Unlike Data Parallelism where different sequences go to different GPUs, in CP the **same sequence** is split across GPUs so that:
- No single GPU needs to hold the entire sequence's activations
- Memory is distributed proportionally to `1/CP_size`
- Enables processing sequences that would otherwise cause OOM

### The Zigzag Load Balancing Problem

With causal attention, each token only attends to its prior tokens. Simply splitting a sequence into CP chunks creates severe **load imbalance**:

```
Naive Split (CP=2):
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 0: tokens [0, 1, 2, 3]     → Attends to: 1+2+3+4 = 10 tokens   │
│  GPU 1: tokens [4, 5, 6, 7]     → Attends to: 5+6+7+8 = 26 tokens   │
│                                                                      │
│  Problem: GPU 1 does 2.6x more work than GPU 0!                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Zigzag Packing Solution

To balance workload, Megatron-LM uses **zigzag packing**: each GPU gets one chunk from the beginning AND one from the end of the sequence:

```
Zigzag Split (CP=2):
┌─────────────────────────────────────────────────────────────────────┐
│  Original sequence: [0, 1, 2, 3, 4, 5, 6, 7]                        │
│                                                                      │
│  Split into 2*CP = 4 chunks:                                        │
│    chunk_0: [0, 1]   chunk_1: [2, 3]   chunk_2: [4, 5]   chunk_3: [6, 7]  │
│                                                                      │
│  Assignment (zigzag pattern):                                        │
│    GPU 0: chunk_0 + chunk_3 = [0, 1, 6, 7]  → workload ~balanced    │
│    GPU 1: chunk_1 + chunk_2 = [2, 3, 4, 5]  → workload ~balanced    │
│                                                                      │
│  Formula: GPU_k gets chunks [k, 2*CP - k - 1]                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Code Flow for Long Sequence Embedding

Here's the complete code flow when using CP for embedding long sequences:

#### Step 1: Data Preparation (hyena_predict_data_step)

```python
# Location: bionemo/evo2/run/predict.py

def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step that handles context parallelism slicing."""

    batch = next(dataloader_iter)

    # Move required tensors to GPU
    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)

    # ⭐ KEY STEP: Slice batch for this CP rank
    # This applies zigzag packing to distribute sequence chunks
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output
```

#### Step 2: Zigzag Slicing (Megatron Core)

```python
# Location: megatron/core/utils.py

def get_batch_on_this_cp_rank(batch, cp_group=None):
    """Slice batch along sequence dimension with zigzag pattern.

    With causal masking, each token only attends to its prior tokens.
    Simply split sequence into CP chunks can result in severe load imbalance.

    We split sequence into 2*CP ranks. Assuming CP=2, we get 4 chunks:
    - chunk_0 and chunk_3 are assigned to GPU0
    - chunk_1 and chunk_2 are assigned to GPU1
    This achieves balanced workload among GPUs in a context parallel group.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    if cp_size > 1:
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2

                # Reshape to [batch, 2*cp_size, seq_len/(2*cp_size), ...]
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1):]
                )

                # Select chunks for this rank: [cp_rank, 2*cp_size - cp_rank - 1]
                index = torch.zeros(2, dtype=torch.int64, device=val.device)
                index[0].fill_(cp_rank)                    # First chunk
                index[1].fill_(2 * cp_size - cp_rank - 1)  # Last chunk (zigzag)

                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                batch[key] = val

    return batch
```

#### Step 3: Forward Pass (Each GPU processes its chunks)

```python
# Location: bionemo/evo2/run/predict.py

def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Forward step - each GPU only processes its assigned chunks."""
    forward_args = {
        "input_ids": batch["tokens"],      # Only this GPU's chunks
        "position_ids": batch["position_ids"],
        "attention_mask": None,
    }
    return model(**forward_args)  # Returns logits for this GPU's chunks
```

#### Step 4: Gather Results Across CP Ranks

```python
# Location: bionemo/evo2/run/predict.py

def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors from all CP ranks and concatenate along sequence dimension.

    ⚠️ WARNING: The gathered output is still in zigzag order!
    To get original sequence order, you need to undo the zigzag packing.
    """
    world_size = parallel_state.get_context_parallel_world_size()

    if world_size == 1:
        return input_

    # Allocate output buffer for all ranks' data
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

    # All-gather across CP group
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(),
        group=parallel_state.get_context_parallel_group()
    )

    # Concatenate along sequence dimension
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class BasePredictor:
    def predict_step(self, batch, batch_idx=None, to_cpu=True):
        """Complete prediction step with gathering across parallelism dimensions."""

        # 1. Run forward pass (each GPU has partial sequence)
        forward_out = self.forward_step(batch)

        if not parallel_state.is_pipeline_last_stage():
            return None

        # 2. Gather across Tensor Parallelism (vocab dimension)
        forward_out_tp_gathered = _gather_along_last_dim(
            forward_out,
            group=parallel_state.get_tensor_model_parallel_group()
        )

        # 3. Gather across Context Parallelism (sequence dimension)
        forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])
        tokens_gathered = _gather_along_cp_dim(batch["tokens"])

        # ⚠️ Note: forward_out_gathered is still in ZIGZAG order!
        # Shape: [batch, seq_len, vocab_size] but seq_len is zigzag-shuffled

        return {"token_logits": forward_out_gathered, ...}
```

### Visual: Complete CP Flow for Embedding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT PARALLEL EMBEDDING FLOW                           │
│                                                                              │
│  Input: Long DNA sequence (e.g., 131072 tokens)                             │
│  Config: CP=4 (4 GPUs for context parallelism)                              │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: ZIGZAG PARTITIONING (in hyena_predict_data_step)
═══════════════════════════════════════════════════════════════════════════════

Original sequence (131072 tokens):
┌─────────────────────────────────────────────────────────────────────────────┐
│ [tok_0 ... tok_16383] [tok_16384 ... tok_32767] ... [tok_114688 ... tok_131071] │
│      chunk_0              chunk_1                        chunk_7                 │
└─────────────────────────────────────────────────────────────────────────────┘

Zigzag assignment (2*CP = 8 chunks):
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU 0: chunk_0 + chunk_7  →  [tok_0...16383] + [tok_114688...131071]       │
│  GPU 1: chunk_1 + chunk_6  →  [tok_16384...32767] + [tok_98304...114687]    │
│  GPU 2: chunk_2 + chunk_5  →  [tok_32768...49151] + [tok_81920...98303]     │
│  GPU 3: chunk_3 + chunk_4  →  [tok_49152...65535] + [tok_65536...81919]     │
│                                                                              │
│  Each GPU processes 32768 tokens (131072 / 4)                               │
│  Memory per GPU: ~1/4 of full sequence activations                          │
└─────────────────────────────────────────────────────────────────────────────┘

Step 2: PARALLEL FORWARD PASS (each GPU independently)
═══════════════════════════════════════════════════════════════════════════════

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    GPU 0     │  │    GPU 1     │  │    GPU 2     │  │    GPU 3     │
│              │  │              │  │              │  │              │
│ chunks 0+7   │  │ chunks 1+6   │  │ chunks 2+5   │  │ chunks 3+4   │
│   ↓          │  │   ↓          │  │   ↓          │  │   ↓          │
│ [Embedding]  │  │ [Embedding]  │  │ [Embedding]  │  │ [Embedding]  │
│   ↓          │  │   ↓          │  │   ↓          │  │   ↓          │
│ [Decoder x N]│  │ [Decoder x N]│  │ [Decoder x N]│  │ [Decoder x N]│
│   ↓          │  │   ↓          │  │   ↓          │  │   ↓          │
│ [LM Head]    │  │ [LM Head]    │  │ [LM Head]    │  │ [LM Head]    │
│   ↓          │  │   ↓          │  │   ↓          │  │   ↓          │
│ logits_0+7   │  │ logits_1+6   │  │ logits_2+5   │  │ logits_3+4   │
│ [32K, V]     │  │ [32K, V]     │  │ [32K, V]     │  │ [32K, V]     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                │                │                │
        └────────────────┴────────────────┴────────────────┘
                                │
                    All-gather across CP group

Step 3: GATHER AND (OPTIONALLY) UNSHUFFLE
═══════════════════════════════════════════════════════════════════════════════

After _gather_along_cp_dim():
┌─────────────────────────────────────────────────────────────────────────────┐
│  Gathered logits shape: [batch, 131072, vocab_size]                         │
│                                                                              │
│  ⚠️ IMPORTANT: Sequence is in ZIGZAG order, not original order!            │
│                                                                              │
│  Zigzag order: [chunk_0, chunk_7, chunk_1, chunk_6, chunk_2, chunk_5, ...]  │
│                                                                              │
│  To get original order, you need to unshuffle:                              │
│    original_order = unshuffle_zigzag(gathered_logits, cp_size=4)            │
└─────────────────────────────────────────────────────────────────────────────┘

Optional: UNSHUFFLE TO ORIGINAL ORDER (if needed for downstream tasks)
═══════════════════════════════════════════════════════════════════════════════

def unshuffle_zigzag(tensor, cp_size, seq_dim=1):
    """Undo zigzag packing to restore original sequence order.

    The zigzag pattern assigns chunks as:
      GPU k gets chunks [k, 2*cp_size - k - 1]

    After gathering, the order is:
      [chunk_0, chunk_7, chunk_1, chunk_6, chunk_2, chunk_5, chunk_3, chunk_4]

    We need to restore:
      [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4, chunk_5, chunk_6, chunk_7]
    """
    # Split into 2*cp_size chunks
    chunks = tensor.chunk(2 * cp_size, dim=seq_dim)

    # Reconstruct original order
    original_chunks = [None] * (2 * cp_size)
    for cp_rank in range(cp_size):
        # Each CP rank contributed 2 chunks
        idx_in_gathered = cp_rank * 2
        chunk_id_first = cp_rank
        chunk_id_second = 2 * cp_size - cp_rank - 1

        original_chunks[chunk_id_first] = chunks[idx_in_gathered]
        original_chunks[chunk_id_second] = chunks[idx_in_gathered + 1]

    return torch.cat(original_chunks, dim=seq_dim)
```

### Sequence Length Requirements for CP

For CP to work correctly, sequences must be divisible by `2 * CP_size`:

```python
# Padding requirement
min_divisor = 2 * context_parallel_size

# Example: CP=4 requires sequences divisible by 8
sequence_length = 131072  # ✓ 131072 / 8 = 16384
sequence_length = 131070  # ✗ Not divisible, needs padding to 131072
```

### Memory Savings with CP

| Sequence Length | CP=1 Memory | CP=2 Memory | CP=4 Memory | CP=8 Memory |
|-----------------|-------------|-------------|-------------|-------------|
| 8,192           | 100%        | ~50%        | ~25%        | ~12.5%      |
| 32,768          | 100%        | ~50%        | ~25%        | ~12.5%      |
| 131,072         | OOM         | OOM         | ~25%        | ~12.5%      |
| 524,288         | OOM         | OOM         | OOM         | ~12.5%      |

*Note: Memory savings are approximate; actual savings depend on model architecture and batch size.*

### Usage Example: Embedding Long Sequences with CP

```bash
# Embed sequences up to 131K tokens using 8 GPUs with CP=4, TP=2
python -m bionemo.evo2.run.predict \
    --fasta long_sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --tensor-parallel-size 2 \
    --context-parallel-size 4 \
    --devices 8 \
    --micro-batch-size 1 \
    --model-size 7b_arc_longcontext
```

### Important Notes for CP Inference

1. **Zigzag Output Order**: The gathered outputs are in zigzag order, not original sequence order. If you need original order for downstream tasks, you must unshuffle.

2. **Per-Token Log Probs**: When using `--output-log-prob-seqs` with `--log-prob-collapse-option per_token` and CP > 1, the per-token values are zigzag-shuffled. A warning is logged:
   ```
   "Per token log probabilities are not supported when using context parallelism.
   The results will be zigzag shuffled along the sequence dimension."
   ```

3. **Sequence Length Padding**: Ensure sequences are padded to be divisible by `2 * CP_size`. Use `--min-length` if needed.

4. **Ring Attention**: The underlying attention mechanism uses Ring Attention to efficiently communicate KV tensors between CP ranks during the forward pass.

### 4. Precision Modes

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRECISION OPTIONS                               │
│                                                                     │
│  Default: BF16-mixed                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  --fp8 (Vortex Style)                                       │   │
│  │  • Only applies FP8 to projection layers in Hyena mixer     │   │
│  │  • Configured via vortex_style_fp8=True in model config     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  --fp8 --full-fp8                                           │   │
│  │  • Full FP8 precision for all eligible layers               │   │
│  │  • Configured via MegatronMixedPrecision plugin            │   │
│  │  • Recipes: delayed, tensorwise, mxfp8, blockwise          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. PEFT Support (LoRA)

The `Evo2LoRA` class enables parameter-efficient fine-tuning:

```python
target_modules = [
    "linear_qkv",           # QKV projections
    "linear_proj",          # Output projections
    "linear_fc1",           # MLP layer 1
    "linear_fc2",           # MLP layer 2
    "short_filter",         # Short convolution filters
    "hyena_filter",         # Hyena layer filters
    "positional_encoding",  # Position encodings
]
```

Load LoRA checkpoint with `--lora-checkpoint-path`.

---

## Inference Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           DETAILED INFERENCE FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────────┘

  1. INITIALIZATION
  ─────────────────
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
  │ parse_args()│────▶│ infer_model_type │────▶│ Select Config Class │
  └─────────────┘     │ (hyena/mamba/    │     │ (HYENA/MAMBA/LLAMA  │
                      │  llama)          │     │  _MODEL_OPTIONS)    │
                      └──────────────────┘     └──────────┬──────────┘
                                                          │
                                                          ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │ Create Config Instance with:                                                        │
  │  • forward_step_fn = hyena_predict_forward_step                                    │
  │  • data_step_fn = hyena_predict_data_step                                          │
  │  • vortex_style_fp8 (if --fp8 and not --full-fp8)                                 │
  │  • hybrid_override_pattern, num_layers (if specified)                             │
  └────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
  2. MODEL SETUP
  ──────────────
  ┌──────────────────┐     ┌────────────────────┐     ┌─────────────────┐
  │ Create Predictor │────▶│  nl.Trainer()      │────▶│  nl.AutoResume  │
  │ (Hyena/Mamba/    │     │  MegatronStrategy  │     │  .setup()       │
  │  Llama)          │     │  MegatronMixed     │     │  Load weights   │
  └──────────────────┘     │  Precision         │     └─────────────────┘
                           └────────────────────┘

  3. DATA PREPARATION
  ───────────────────
  ┌─────────────────────┐     ┌───────────────────────┐
  │ SimpleFastaDataset  │────▶│ PredictDataModule     │
  │  • Load FASTA       │     │  • WrappedDataLoader  │
  │  • Tokenize (byte)  │     │  • padding_collate_fn │
  │  • Add BOS (opt)    │     └───────────────────────┘
  │  • Create loss_mask │
  └─────────────────────┘

  4. PREDICTION LOOP
  ──────────────────
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ trainer.predict(model, datamodule)                                               │
  │                                                                                  │
  │   For each batch:                                                                │
  │   ┌────────────────────────────────────────────────────────────────────────────┐│
  │   │ 1. data_step(): Prepare batch, apply CP slicing                            ││
  │   │                                                                            ││
  │   │ 2. forward_step(): Run model forward                                       ││
  │   │    model(input_ids, position_ids, attention_mask=None)                     ││
  │   │                                                                            ││
  │   │ 3. predict_step(): Post-process                                            ││
  │   │    • Gather across TP: _gather_along_last_dim()                           ││
  │   │    • Gather across CP: _gather_along_cp_dim()                             ││
  │   │    • Compute log_probs or return logits                                   ││
  │   │                                                                            ││
  │   │ 4. PredictionWriter: Save results                                          ││
  │   └────────────────────────────────────────────────────────────────────────────┘│
  └──────────────────────────────────────────────────────────────────────────────────┘

  5. OUTPUT
  ─────────
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ output_dir/                                                                      │
  │  ├── seq_idx_map.json           # Sequence ID to index mapping                  │
  │  ├── predictions_0.pt           # Batch predictions (if write_interval=batch)  │
  │  └── predictions_epoch_0.pt     # Epoch predictions (if write_interval=epoch)  │
  │                                                                                  │
  │ Each prediction contains:                                                        │
  │  • token_logits: [batch, seq_len, vocab_size] OR                               │
  │  • log_probs_seqs: [batch] or [batch, seq_len] (if output_log_prob_seqs)       │
  │  • pad_mask / loss_mask                                                         │
  │  • seq_idx                                                                       │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dissecting the Checkpoint

### Tools for Checkpoint Analysis

#### 1. Convert NeMo2 to HuggingFace
```bash
python -m bionemo.evo2.utils.checkpoint.nemo2_to_hf \
    --model-type llama \
    --model-path /path/to/nemo_ckpt \
    --output-dir /path/to/hf_output
```

#### 2. Remove Optimizer States
```bash
python -m bionemo.evo2.utils.checkpoint.evo2_remove_optimizer \
    --input-dir /path/to/full_ckpt \
    --output-dir /path/to/weights_only
```

#### 3. Convert Tensor Parallelism
```bash
python -m bionemo.evo2.utils.checkpoint.convert_checkpoint_model_parallel_evo2 \
    --input-ckpt /path/to/ckpt_tp1 \
    --output-ckpt /path/to/ckpt_tp4 \
    --target-tp 4
```

### Loading and Inspecting Checkpoint

```python
import torch
from pathlib import Path

# For torch_dist format
ckpt_path = Path("/path/to/checkpoint/weights/__0_0.distcp/__0_0.pt")
state_dict = torch.load(ckpt_path, map_location="cpu")

# List all parameter keys
for key in sorted(state_dict.keys()):
    tensor = state_dict[key]
    print(f"{key}: {tensor.shape}, dtype={tensor.dtype}")

# Key groups:
# - model.embedding.*: Token embeddings
# - model.decoder.layers.*.mixer.*: Hyena/Mamba/Attention layers
# - model.decoder.layers.*.mlp.*: Feed-forward layers
# - model.decoder.layers.*_layernorm.*: Layer normalizations
# - model.decoder.final_layernorm.*: Final layer norm
# - model.output_layer.*: LM head (may be tied to embeddings)
```

### Memory Estimation

For a 7B Evo2 model:
- BF16: ~14 GB (7B × 2 bytes)
- FP8: ~7 GB (7B × 1 byte)
- With optimizer states (training): ~56 GB (Adam: 4× model size)

---

## Key Classes and Their Responsibilities

| Class | Location | Responsibility |
|-------|----------|----------------|
| `SimpleFastaDataset` | `data/fasta_dataset.py` | Load FASTA, tokenize, create batches |
| `PredictDataModule` | `run/predict.py` | PyTorch Lightning DataModule wrapper |
| `BasePredictor` | `run/predict.py` | Common prediction logic, output gathering |
| `HyenaPredictor` | `run/predict.py` | Hyena-specific prediction wrapper |
| `MambaPredictor` | `run/predict.py` | Mamba-specific prediction wrapper |
| `LlamaPredictor` | `run/predict.py` | Llama/Eden prediction wrapper |
| `Evo2LoRA` | `models/peft.py` | LoRA adapter for Evo2 |
| `MambaModel` | `models/mamba.py` | Mamba model NeMo wrapper |
| `Evo2StyleMCoreMambaModel` | `models/mamba.py` | Custom MCore Mamba with Evo2 loss |
| `PredictionWriter` | `bionemo.llm.utils.callbacks` | Save predictions to disk |

---

## Common Usage Patterns

### Basic Inference
```bash
python -m bionemo.evo2.run.predict \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./predictions \
    --model-size 7b_arc_longcontext
```

### With Tensor Parallelism
```bash
python -m bionemo.evo2.run.predict \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./predictions \
    --tensor-parallel-size 4 \
    --devices 4
```

### With FP8 Precision
```bash
python -m bionemo.evo2.run.predict \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./predictions \
    --fp8 --full-fp8 \
    --fp8-recipe tensorwise
```

### Compute Log Probabilities
```bash
python -m bionemo.evo2.run.predict \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./predictions \
    --output-log-prob-seqs \
    --log-prob-collapse-option mean
```

### Text Generation
```bash
python -m bionemo.evo2.run.infer \
    --prompt "|d__Bacteria;p__Pseudomonadota;..." \
    --ckpt-dir /path/to/evo2-7b \
    --max-new-tokens 1024 \
    --temperature 1.0 \
    --output-file generated.txt
```

---

## References

- [predict.py](src/bionemo/evo2/run/predict.py) - Main prediction entry point
- [infer.py](src/bionemo/evo2/run/infer.py) - Generation entry point
- [mamba.py](src/bionemo/evo2/models/mamba.py) - Mamba model implementation
- [llama.py](src/bionemo/evo2/models/llama.py) - Eden/Llama model configs
- [peft.py](src/bionemo/evo2/models/peft.py) - LoRA implementation
- [fasta_dataset.py](src/bionemo/evo2/data/fasta_dataset.py) - FASTA data loading
- [params.py](src/bionemo/evo2/utils/checkpoint/params.py) - Checkpoint parameter specs
