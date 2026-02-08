# Evo2 Embedding Extraction - Design Document

## 1. Overview

This document describes the architecture for extracting embeddings from Evo2 models. The solution enables:
- **Partial checkpoint loading**: Load only the first N layers of a pretrained model
- **Hidden state extraction**: Return decoder hidden states instead of logits
- **Flexible pooling**: Apply configurable pooling operations (mean, max, last, first)
- **Full parallelism support**: Context Parallelism (CP) and Tensor Parallelism (TP) for long sequences and large batches

### 1.1 Design Principles

1. **Minimal changes**: Extend existing classes rather than modify them
2. **Reuse upstream**: Leverage NeMo/Megatron infrastructure for checkpoint loading, parallelism, data handling
3. **Composition over inheritance**: Use mixins and wrappers where possible
4. **Configuration-driven**: Control behavior via existing config patterns

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           EVO2 EMBEDDING EXTRACTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 USER INTERFACE                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              CLI: embed.py                                        │   │
│  │  python -m bionemo.evo2.run.embed \                                              │   │
│  │      --fasta input.fa \                                                          │   │
│  │      --ckpt-dir /path/to/evo2-7b \                                               │   │
│  │      --output-dir ./embeddings \                                                 │   │
│  │      --embedding-layer 16 \              # Extract from layer 16 (0=embedding)   │   │
│  │      --pooling-strategy mean \           # mean|max|last|first|per_token         │   │
│  │      --context-parallel-size 4 \         # CP for long sequences                 │   │
│  │      --tensor-parallel-size 2            # TP for model parallelism              │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER (REUSED)                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    SimpleFastaDataset (existing)                                  │   │
│  │  • NvFaidx for efficient FASTA reading                                           │   │
│  │  • Byte-level tokenization                                                       │   │
│  │  • Returns: {tokens, position_ids, loss_mask, seq_idx}                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                             │                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    PredictDataModule (existing)                                   │   │
│  │  • Batching with padding_collate_fn                                              │   │
│  │  • Variable sequence length support                                              │   │
│  │  • min_length padding for CP divisibility                                        │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL LAYER (NEW + REUSED)                                  │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                     EmbeddingExtractor (NEW)                                      │   │
│  │  Mixin class providing embedding extraction logic:                                │   │
│  │  • extract_embeddings(): Runs forward, applies pooling                           │   │
│  │  • _pool_hidden_states(): Implements pooling strategies                          │   │
│  │  • _gather_hidden_states(): Gathers across TP/CP ranks                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                             │                                            │
│                            ┌────────────────┼────────────────┐                          │
│                            ▼                ▼                ▼                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                      │
│  │ HyenaEmbedder    │  │ MambaEmbedder    │  │ LlamaEmbedder    │                      │
│  │ (NEW)            │  │ (NEW)            │  │ (NEW)            │                      │
│  │                  │  │                  │  │                  │                      │
│  │ EmbeddingMixin + │  │ EmbeddingMixin + │  │ EmbeddingMixin + │                      │
│  │ HyenaModel       │  │ MambaModel       │  │ GPTModel         │                      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘                      │
│           │                     │                     │                                 │
│           └─────────────────────┼─────────────────────┘                                 │
│                                 │                                                       │
│  ┌──────────────────────────────▼───────────────────────────────────────────────────┐   │
│  │                     TruncatedModelWrapper (NEW)                                   │   │
│  │  Wraps the underlying Megatron model to:                                         │   │
│  │  • Skip output_layer (LM head)                                                   │   │
│  │  • Optionally truncate decoder.layers to first N layers                          │   │
│  │  • Return hidden_states instead of logits                                        │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FORWARD PASS (MODIFIED)                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                  embedding_forward_step() (NEW)                                   │   │
│  │                                                                                   │   │
│  │  Similar to hyena_predict_forward_step but:                                      │   │
│  │  • Calls model.forward_for_embeddings() instead of model()                       │   │
│  │  • Returns hidden_states tensor [batch, seq_len, hidden_size]                    │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │               Truncated Forward Flow (inside Megatron model)                      │   │
│  │                                                                                   │   │
│  │    ┌─────────────────┐                                                           │   │
│  │    │   Embedding     │  word_embeddings [vocab_size=512, hidden_size]            │   │
│  │    │   + LayerNorm   │  (Layer 0 conceptually)                                   │   │
│  │    └────────┬────────┘                                                           │   │
│  │             │                                                                     │   │
│  │             ▼                                                                     │   │
│  │    ┌─────────────────────────────────────────────────────────────────────┐       │   │
│  │    │         Decoder Stack (TRUNCATED to embedding_layer)                │       │   │
│  │    │  Layer 1  ──▶  Layer 2  ──▶  ...  ──▶  Layer N                     │       │   │
│  │    │                                        (embedding_layer)            │       │   │
│  │    │                                              │                      │       │   │
│  │    │                                              ▼                      │       │   │
│  │    │                                   [hidden_states output]            │       │   │
│  │    │                                                                     │       │   │
│  │    │  ╳ Layers N+1 to num_layers (SKIPPED - not loaded)                 │       │   │
│  │    └─────────────────────────────────────────────────────────────────────┘       │   │
│  │             │                                                                     │   │
│  │             ▼                                                                     │   │
│  │    ┌─────────────────┐                                                           │   │
│  │    │   Final Norm    │  Applied if post_process=True                             │   │
│  │    └────────┬────────┘                                                           │   │
│  │             │                                                                     │   │
│  │             ▼                                                                     │   │
│  │    ╳ OUTPUT LAYER (LM HEAD) - SKIPPED                                            │   │
│  │                                                                                   │   │
│  │    Return: hidden_states [seq_len, batch, hidden_size]                           │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING STEP (NEW)                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    EmbeddingExtractor.predict_step()                              │   │
│  │                                                                                   │   │
│  │   1. Forward pass → hidden_states [seq, batch, hidden]                           │   │
│  │   2. Transpose → [batch, seq, hidden]                                            │   │
│  │   3. Gather across TP: _gather_along_last_dim() (if TP > 1)                      │   │
│  │   4. Gather across CP: _gather_along_cp_dim()                                    │   │
│  │   5. Apply pooling based on strategy:                                            │   │
│  │                                                                                   │   │
│  │      ┌─────────────────────────────────────────────────────────────────────┐     │   │
│  │      │  Pooling Strategies                                                 │     │   │
│  │      │  ───────────────────                                                │     │   │
│  │      │  • "mean":      mean(hidden * mask) / sum(mask)  → [B, H]          │     │   │
│  │      │  • "max":       max(hidden * mask)               → [B, H]          │     │   │
│  │      │  • "last":      hidden[:, last_valid_idx, :]     → [B, H]          │     │   │
│  │      │  • "first":     hidden[:, 0, :]                  → [B, H]          │     │   │
│  │      │  • "per_token": hidden * mask                    → [B, S, H]       │     │   │
│  │      └─────────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                                   │   │
│  │   6. Return: {embeddings, seq_idx, [pad_mask if per_token]}                      │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER (REUSED)                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    PredictionWriter callback (existing)                           │   │
│  │                                                                                   │   │
│  │  Writes to output_dir/:                                                          │   │
│  │  • embeddings_batch_*.pt   - Embedding tensors                                   │   │
│  │  • seq_idx_map.json        - Maps batch index to original sequence ID            │   │
│  │                                                                                   │   │
│  │  Output shapes:                                                                  │   │
│  │  • Pooled: [batch_size, hidden_size]                                             │   │
│  │  • Per-token: [batch_size, seq_len, hidden_size] + pad_mask                      │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  CLASS HIERARCHY                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────────────────────┐
                           │  LightningPassthrough           │
                           │  PredictionMixin (NeMo)         │
                           │  ───────────────────────────    │
                           │  + forward_step()               │
                           │  + data_step()                  │
                           └────────────────┬────────────────┘
                                            │ inherits
                           ┌────────────────▼────────────────┐
                           │       BasePredictor             │
                           │       (existing)                │
                           │  ───────────────────────────    │
                           │  + output_log_prob_seqs: bool   │
                           │  + log_prob_collapse_option     │
                           │  ───────────────────────────    │
                           │  + predict_step()               │
                           │  # returns logits/log_probs     │
                           └────────────────┬────────────────┘
                                            │ inherits
                           ┌────────────────▼────────────────┐
                           │     EmbeddingExtractorMixin     │
                           │           (NEW)                 │
                           │  ───────────────────────────    │
                           │  + embedding_layer: int         │
                           │  + pooling_strategy: str        │
                           │  + include_final_norm: bool     │
                           │  ───────────────────────────    │
                           │  + predict_step() [override]    │
                           │  + _pool_hidden_states()        │
                           │  + _apply_mask()                │
                           │  + forward_for_embeddings()     │
                           └────────────────┬────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
┌─────────────▼─────────────┐ ┌─────────────▼─────────────┐ ┌─────────────▼─────────────┐
│     HyenaEmbedder         │ │     MambaEmbedder         │ │     LlamaEmbedder         │
│         (NEW)             │ │         (NEW)             │ │         (NEW)             │
│  ─────────────────────    │ │  ─────────────────────    │ │  ─────────────────────    │
│  EmbeddingExtractorMixin  │ │  EmbeddingExtractorMixin  │ │  EmbeddingExtractorMixin  │
│         +                 │ │         +                 │ │         +                 │
│     HyenaModel            │ │     MambaModel            │ │      GPTModel             │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            MEGATRON MODEL STRUCTURE                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            HyenaModel (Megatron-Core)                                    │
│  ─────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                          │
│   Attributes:                                                                            │
│   ───────────                                                                            │
│   + transformer_config: TransformerConfig                                               │
│   + vocab_size: int                                                                     │
│   + pre_process: bool        # True if pipeline first stage (has embedding)            │
│   + post_process: bool       # True if pipeline last stage (has output_layer)          │
│   + embedding: LanguageModelEmbedding                                                   │
│   + decoder: HyenaStack      # Contains layers ModuleList                               │
│   + output_layer: ColumnParallelLinear  # LM head - SKIP FOR EMBEDDINGS                │
│                                                                                          │
│   Key insight for embedding extraction:                                                 │
│   ─────────────────────────────────────                                                 │
│   In forward(), when labels=None:                                                       │
│     - Runs: embedding → decoder → output_layer → returns logits                         │
│                                                                                          │
│   For embeddings, we need to:                                                           │
│     - Run: embedding → decoder (truncated) → return hidden_states                       │
│     - Skip output_layer entirely                                                        │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                HyenaStack (decoder)                                      │
│  ─────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                          │
│   + layers: nn.ModuleList[HyenaLayer | TransformerLayer]                                │
│   + final_norm: TENorm (if post_layer_norm=True)                                        │
│   + num_layers_per_pipeline_rank: int                                                   │
│                                                                                          │
│   forward(hidden_states, ...) → hidden_states                                           │
│     - Iterates through self.layers                                                      │
│     - Applies final_norm if post_process and post_layer_norm                            │
│                                                                                          │
│   For truncation strategy (Option A - config-based):                                    │
│   ──────────────────────────────────────────────────                                    │
│   Use config.num_layers to control how many layers are instantiated.                    │
│   Checkpoint loading will only load weights for instantiated layers.                    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Component Design

### 4.1 EmbeddingExtractorMixin

This mixin provides embedding extraction functionality. It overrides `predict_step()` from `BasePredictor`.

```python
# Pseudocode for implementation reference

class EmbeddingExtractorMixin:
    """Mixin that provides embedding extraction capabilities.

    Must be used with a model class that has:
    - self.module (the Megatron model with embedding, decoder, output_layer)
    - self.tokenizer
    """

    def __init__(
        self,
        *args,
        embedding_layer: int | None = None,  # None = use all layers
        pooling_strategy: Literal["mean", "max", "last", "first", "per_token"] = "mean",
        include_final_norm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.pooling_strategy = pooling_strategy
        self.include_final_norm = include_final_norm

    def predict_step(self, batch, batch_idx: int | None = None) -> dict[str, Tensor]:
        """Extract embeddings from the model.

        Returns:
            dict with keys:
            - "embeddings": Tensor of shape [B, H] or [B, S, H] for per_token
            - "seq_idx": Tensor mapping to original sequence indices
            - "pad_mask": (only for per_token) Boolean mask [B, S]
        """
        if len(batch) == 0:
            return None

        with torch.no_grad():
            # 1. Get hidden states from truncated forward pass
            hidden_states = self.forward_for_embeddings(batch)

        if not parallel_state.is_pipeline_last_stage():
            return None

        # 2. Gather across TP (hidden dimension is split across TP ranks)
        # Note: For HyenaModel, hidden_size is NOT split across TP, but we keep
        # this for compatibility with models that do split
        hidden_gathered = hidden_states  # May need TP gather for some models

        # 3. Gather across CP (sequence dimension is split across CP ranks)
        hidden_gathered = _gather_along_cp_dim(hidden_gathered, seq_dim=1)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])

        # 4. Apply pooling
        embeddings = self._pool_hidden_states(
            hidden_gathered,
            loss_mask_gathered,
            self.pooling_strategy
        )

        result = {
            "embeddings": embeddings.cpu(),
            "seq_idx": batch["seq_idx"].cpu(),
        }

        if self.pooling_strategy == "per_token":
            result["pad_mask"] = loss_mask_gathered.cpu()

        return result

    def forward_for_embeddings(self, batch) -> Tensor:
        """Run forward pass and return hidden states instead of logits.

        This method:
        1. Runs embedding layer
        2. Runs decoder (truncated to embedding_layer if specified)
        3. Optionally applies final norm
        4. SKIPS output_layer (LM head)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Access the underlying Megatron model
        model = self.module

        input_ids = batch["tokens"]
        position_ids = batch["position_ids"]

        # Run embedding
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)

        # Run decoder (all instantiated layers)
        hidden_states = model.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            # ... other args
        )

        # Apply final norm if requested and available
        if self.include_final_norm and hasattr(model.decoder, 'final_norm'):
            hidden_states = model.decoder.final_norm(hidden_states)

        # Transpose from [seq, batch, hidden] to [batch, seq, hidden]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        return hidden_states

    def _pool_hidden_states(
        self,
        hidden_states: Tensor,  # [B, S, H]
        mask: Tensor,           # [B, S] boolean or float
        strategy: str,
    ) -> Tensor:
        """Apply pooling strategy to hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            mask: [batch, seq_len] - True/1.0 for valid tokens
            strategy: One of "mean", "max", "last", "first", "per_token"

        Returns:
            Pooled tensor:
            - [batch, hidden_size] for mean/max/last/first
            - [batch, seq_len, hidden_size] for per_token
        """
        mask_float = mask.float().unsqueeze(-1)  # [B, S, 1]

        if strategy == "per_token":
            # Return all hidden states, masked
            return hidden_states * mask_float

        elif strategy == "mean":
            # Mean of valid tokens
            masked_sum = (hidden_states * mask_float).sum(dim=1)  # [B, H]
            valid_counts = mask_float.sum(dim=1).clamp(min=1.0)   # [B, 1]
            return masked_sum / valid_counts

        elif strategy == "max":
            # Max pooling over valid tokens
            # Set invalid positions to -inf before max
            masked_hidden = hidden_states.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return masked_hidden.max(dim=1).values  # [B, H]

        elif strategy == "last":
            # Get last valid token for each sequence
            # Find the index of the last True in mask for each batch item
            seq_lengths = mask.sum(dim=1).long() - 1  # [B]
            seq_lengths = seq_lengths.clamp(min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths]  # [B, H]

        elif strategy == "first":
            # Simply take the first token (position 0)
            return hidden_states[:, 0, :]  # [B, H]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
```

### 4.2 Configuration Strategy for Layer Truncation

**Key Insight**: The existing config system already supports `num_layers` override!

From `predict.py`:
```python
if num_layers is not None:
    config_modifiers_init["num_layers"] = num_layers
```

This means we can control how many layers are instantiated via config. The checkpoint loader (`ckpt_load_strictness="log_all"`) will only load weights for layers that exist.

**Strategy**:
- Use `num_layers` config parameter to instantiate only the layers we need
- Set `embedding_layer = num_layers` in our embedder
- The model will be created with exactly `embedding_layer` decoder layers
- Checkpoint loading will load only those layers (others are simply not instantiated)

### 4.3 New Forward Step Function

```python
def embedding_forward_step(model, batch) -> torch.Tensor:
    """Forward step that returns hidden states for embedding extraction.

    Unlike hyena_predict_forward_step which returns logits, this returns
    the hidden states before the output layer.
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": None,
    }

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    # Call our custom forward that skips the LM head
    return model.forward_for_embeddings(**forward_args)
```

---

## 5. Data Flow with Context Parallelism

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING EXTRACTION WITH CONTEXT PARALLELISM                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Input: Sequence of 16 tokens, CP=2
Tokens: [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15]

Step 1: Zigzag Split (in hyena_predict_data_step via get_batch_on_this_cp_rank)
─────────────────────────────────────────────────────────────────────────────────
  CP Rank 0 gets: [T0, T1, T2, T3, T12, T13, T14, T15]  (chunks 0, 3)
  CP Rank 1 gets: [T4, T5, T6, T7, T8, T9, T10, T11]    (chunks 1, 2)

Step 2: Independent Forward Pass on Each CP Rank
─────────────────────────────────────────────────
  CP Rank 0: Embedding → Decoder → hidden_states_0 [8, B, H]
  CP Rank 1: Embedding → Decoder → hidden_states_1 [8, B, H]

Step 3: Gather Across CP Ranks (_gather_along_cp_dim)
─────────────────────────────────────────────────────
  All-gather: [hidden_states_0, hidden_states_1]
  Concatenate along seq dim → [16, B, H]

  ⚠️ OUTPUT IS IN ZIGZAG ORDER:
  [H0, H1, H2, H3, H12, H13, H14, H15, H4, H5, H6, H7, H8, H9, H10, H11]
      └── CP rank 0 ──┘                   └── CP rank 1 ──┘

Step 4: Pooling (handles zigzag transparently for most strategies)
─────────────────────────────────────────────────────────────────
  For "mean" pooling:
    - Sum all valid hidden states (order doesn't matter)
    - Divide by count
    - Result: [B, H] embedding vector

  For "max" pooling:
    - Take element-wise max (order doesn't matter)
    - Result: [B, H] embedding vector

  For "last" pooling:
    - ⚠️ REQUIRES UNSHUFFLING to find true last token
    - Or use mask to identify last valid token position

  For "per_token" pooling:
    - ⚠️ RETURNS ZIGZAG ORDER - user must unshuffle if needed
    - Return: [B, S, H] with warning about ordering

Step 5: Output
──────────────
  embeddings: [B, H] or [B, S, H]
  seq_idx: Maps to original FASTA sequences
```

### 5.1 Handling Zigzag for Position-Sensitive Pooling

For `last` and `per_token` strategies with CP > 1:

```python
def _unshuffle_zigzag(tensor: Tensor, cp_size: int, seq_dim: int = 1) -> Tensor:
    """Restore original sequence order from zigzag-packed tensor.

    After CP gather, sequences are in zigzag order:
    [chunk_0, chunk_3, chunk_1, chunk_2] for CP=2

    This function restores to:
    [chunk_0, chunk_1, chunk_2, chunk_3]
    """
    if cp_size == 1:
        return tensor

    num_chunks = 2 * cp_size
    chunks = tensor.chunk(num_chunks, dim=seq_dim)

    # Reconstruct original order
    # Zigzag pattern: rank r gets chunks [r*2, num_chunks - 1 - r*2]
    original_order = [None] * num_chunks
    chunk_idx = 0
    for rank in range(cp_size):
        original_order[rank] = chunks[chunk_idx]
        chunk_idx += 1
        original_order[num_chunks - 1 - rank] = chunks[chunk_idx]
        chunk_idx += 1

    return torch.cat(original_order, dim=seq_dim)
```

---

## 6. File Structure

```
sub-packages/bionemo-evo2/
├── src/bionemo/evo2/
│   ├── run/
│   │   ├── predict.py          # Existing - DO NOT MODIFY
│   │   └── embed.py            # NEW - CLI entry point for embeddings
│   └── models/
│       ├── __init__.py         # Update exports
│       └── embedder.py         # NEW - EmbeddingExtractorMixin + Embedder classes
└── docs/
    └── evo2_embedding_design.md  # This document
```

---

## 7. API Design

### 7.1 Command Line Interface

```bash
# Basic embedding extraction
python -m bionemo.evo2.run.embed \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --pooling-strategy mean

# Extract from specific layer (layer 16 of 32)
python -m bionemo.evo2.run.embed \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --embedding-layer 16 \
    --pooling-strategy mean

# With Context Parallelism for long sequences
python -m bionemo.evo2.run.embed \
    --fasta long_sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --context-parallel-size 4 \
    --tensor-parallel-size 2 \
    --devices 8 \
    --min-length 131072 \
    --pooling-strategy mean

# Per-token embeddings (for downstream tasks needing position info)
python -m bionemo.evo2.run.embed \
    --fasta sequences.fa \
    --ckpt-dir /path/to/evo2-7b \
    --output-dir ./embeddings \
    --pooling-strategy per_token
```

### 7.2 New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding-layer` | int | None | Layer to extract embeddings from (0=embedding, N=after layer N). None uses all layers. |
| `--pooling-strategy` | str | "mean" | Pooling strategy: mean, max, last, first, per_token |
| `--no-final-norm` | flag | False | Skip final layer norm (use raw hidden states) |

### 7.3 Programmatic API

```python
from bionemo.evo2.run.embed import extract_embeddings

# Simple usage
embeddings = extract_embeddings(
    fasta_path="sequences.fa",
    ckpt_dir="/path/to/evo2-7b",
    output_dir="./embeddings",
    pooling_strategy="mean",
)

# Advanced usage with partial model
embeddings = extract_embeddings(
    fasta_path="sequences.fa",
    ckpt_dir="/path/to/evo2-7b",
    output_dir="./embeddings",
    embedding_layer=16,           # Use only first 16 layers
    pooling_strategy="mean",
    context_parallel_size=4,
    tensor_parallel_size=2,
    devices=8,
    micro_batch_size=4,
)
```

---

## 8. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `embedder.py` with `EmbeddingExtractorMixin`
- [ ] Implement `HyenaEmbedder`, `MambaEmbedder`, `LlamaEmbedder` classes
- [ ] Implement `forward_for_embeddings()` method
- [ ] Implement `_pool_hidden_states()` with all strategies
- [ ] Implement `embedding_forward_step()` function

### Phase 2: CLI and Integration
- [ ] Create `embed.py` CLI entry point
- [ ] Add argument parsing (reuse from predict.py where possible)
- [ ] Integrate with existing `PredictDataModule`
- [ ] Configure `PredictionWriter` for embedding outputs

### Phase 3: Testing
- [ ] Unit tests for pooling strategies
- [ ] Unit tests for zigzag unshuffling
- [ ] Integration test with small model
- [ ] Integration test with CP > 1
- [ ] Test partial layer loading

### Phase 4: Documentation
- [ ] Update README with embedding extraction examples
- [ ] Add docstrings to all public methods
- [ ] Create usage notebook

---

## 9. Key Design Decisions

### 9.1 Why Mixin over Subclassing?

A mixin approach (`EmbeddingExtractorMixin`) is preferred because:
1. **Single point of change**: Embedding logic is in one place
2. **Easy to extend**: Can be added to any model class
3. **Testable**: Can test the mixin independently

### 9.2 Why Config-based Truncation?

Using `num_layers` in config rather than runtime truncation because:
1. **Memory efficient**: Only instantiate needed layers
2. **Already supported**: Config system handles this
3. **Checkpoint compatible**: Loader handles missing keys gracefully

### 9.3 Why Not Modify Upstream?

We avoid modifying `BasePredictor` or NeMo/Megatron code because:
1. **Maintainability**: Easier to update when upstream changes
2. **Separation of concerns**: Embedding is a specialized use case
3. **No conflicts**: Our code is purely additive

---

## 10. Appendix: Memory Considerations

### 10.1 GPU Memory for Embedding Extraction

Embedding extraction uses **less memory** than full inference because:
- No output layer weights (vocab_size × hidden_size)
- Optionally fewer decoder layers
- No logits tensor (batch × seq × vocab_size)

Approximate memory savings for 7B model:
- Full model: ~14GB (bf16)
- With embedding_layer=16 (half layers): ~8GB (bf16)
- No output layer: Saves ~4MB (512 × 4096 × 2 bytes)

### 10.2 Batch Size Recommendations

| Sequence Length | CP Size | Recommended Batch Size | Memory per GPU |
|-----------------|---------|------------------------|----------------|
| 8K              | 1       | 4-8                    | ~16GB          |
| 32K             | 2       | 2-4                    | ~20GB          |
| 131K            | 4       | 1-2                    | ~24GB          |
| 512K            | 8       | 1                      | ~32GB          |
