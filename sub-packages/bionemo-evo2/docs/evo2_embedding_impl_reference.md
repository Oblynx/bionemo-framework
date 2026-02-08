# Evo2 Embedding Implementation Reference

This is a companion document to `evo2_embedding_design.md` providing additional implementation details for developers.

## 1. Inheritance Diagram

```
                                    ┌──────────────────────────┐
                                    │   torch.nn.Module        │
                                    └────────────┬─────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         │                                               │
           ┌─────────────▼─────────────┐               ┌─────────────────▼───────────────┐
           │   MegatronModule          │               │   LightningModule               │
           │   (megatron.core)         │               │   (pytorch_lightning)           │
           └─────────────┬─────────────┘               └─────────────────┬───────────────┘
                         │                                               │
           ┌─────────────▼─────────────┐               ┌─────────────────▼───────────────┐
           │   LanguageModule          │               │   LightningPassthrough          │
           │   (megatron.core)         │               │   PredictionMixin (NeMo)        │
           │                           │               │                                 │
           │   + embedding             │               │   + forward_step()              │
           │   + decoder               │               │   + data_step()                 │
           │   + output_layer          │               │   + predict_step()              │
           └─────────────┬─────────────┘               └─────────────────┬───────────────┘
                         │                                               │
           ┌─────────────▼─────────────┐                                 │
           │   HyenaModel (MCoreHyena) │                                 │
           │   (nemo.collections.llm)  │                                 │
           │                           │                                 │
           │   + forward()             │                                 │
           │   + set_input_tensor()    │                                 │
           └─────────────┬─────────────┘                                 │
                         │                                               │
                         │              ┌────────────────────────────────┘
                         │              │
                         │    ┌─────────▼─────────┐
                         │    │   BasePredictor   │
                         │    │   (bionemo)       │
                         │    │   [existing]      │
                         │    └─────────┬─────────┘
                         │              │
                         │    ┌─────────▼──────────────────┐
                         │    │   EmbeddingExtractorMixin  │
                         │    │   (bionemo) [NEW]          │
                         │    │                            │
                         │    │   + embedding_layer        │
                         │    │   + pooling_strategy       │
                         │    │   + predict_step()         │
                         │    │   + forward_for_embed..()  │
                         │    │   + _pool_hidden_states()  │
                         │    └─────────┬──────────────────┘
                         │              │
                         │              │  (MRO combines both)
                         └──────────────┼──────────────────┐
                                        │                  │
                         ┌──────────────▼────────────┐     │
                         │      HyenaEmbedder        │     │
                         │      (bionemo) [NEW]      │     │
                         │                           │     │
                         │  class HyenaEmbedder(     │     │
                         │    EmbeddingExtractorMixin│     │
                         │    HyenaModel             │◄────┘
                         │  ):                       │
                         │      pass                 │
                         └───────────────────────────┘
```

## 2. Method Resolution Order (MRO) Explanation

Python's MRO for `HyenaEmbedder`:

```python
class HyenaEmbedder(EmbeddingExtractorMixin, HyenaModel):
    pass

# MRO will be:
# HyenaEmbedder
# └─ EmbeddingExtractorMixin  (our mixin - gets priority for predict_step)
#    └─ BasePredictor         (existing predictor base)
#       └─ LightningPassthroughPredictionMixin
# └─ HyenaModel               (NeMo wrapper)
#    └─ GPTModel
#       └─ MCoreHyenaModel (via self.module)
#          └─ LanguageModule
#             └─ MegatronModule
```

**Key insight**: By placing `EmbeddingExtractorMixin` first in the inheritance list, its `predict_step()` method takes precedence over `BasePredictor.predict_step()`.

## 3. Sequence Diagram: Embedding Extraction

```
┌─────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  User   │     │  embed.py   │     │ HyenaEmbedder   │     │ HyenaModel   │     │ HyenaStack   │
│  (CLI)  │     │  (entry)    │     │ (our class)     │     │ (NeMo)       │     │ (decoder)    │
└────┬────┘     └──────┬──────┘     └────────┬────────┘     └──────┬───────┘     └──────┬───────┘
     │                 │                     │                     │                    │
     │  Run embed      │                     │                     │                    │
     │────────────────►│                     │                     │                    │
     │                 │                     │                     │                    │
     │                 │  Create embedder    │                     │                    │
     │                 │  with config        │                     │                    │
     │                 │  (num_layers=N)     │                     │                    │
     │                 │────────────────────►│                     │                    │
     │                 │                     │                     │                    │
     │                 │                     │  __init__ calls     │                    │
     │                 │                     │  HyenaModel.__init__│                    │
     │                 │                     │────────────────────►│                    │
     │                 │                     │                     │                    │
     │                 │                     │                     │  Build decoder    │
     │                 │                     │                     │  with N layers    │
     │                 │                     │                     │─────────────────►│
     │                 │                     │                     │                    │
     │                 │  trainer.predict()  │                     │                    │
     │                 │────────────────────►│                     │                    │
     │                 │                     │                     │                    │
     │                 │                     │  predict_step(batch)│                    │
     │                 │                     │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─│                    │
     │                 │                     │                     │                    │
     │                 │                     │  forward_for_       │                    │
     │                 │                     │  embeddings()       │                    │
     │                 │                     │─ ─ ─ ─ ─ ─ ─ ─ ─ ─►│                    │
     │                 │                     │                     │                    │
     │                 │                     │                     │  self.embedding()  │
     │                 │                     │                     │─ ─ ─ ─ ─ ─ ─ ─ ─►│
     │                 │                     │                     │◄─ ─ ─ ─ ─ ─ ─ ─ ─│
     │                 │                     │                     │                    │
     │                 │                     │                     │  self.decoder()   │
     │                 │                     │                     │  (N layers only)  │
     │                 │                     │                     │─────────────────►│
     │                 │                     │                     │                    │
     │                 │                     │                     │  Layer 1          │
     │                 │                     │                     │  ...              │
     │                 │                     │                     │  Layer N          │
     │                 │                     │                     │  final_norm       │
     │                 │                     │                     │◄─────────────────│
     │                 │                     │                     │                    │
     │                 │                     │  hidden_states      │                    │
     │                 │                     │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─│                    │
     │                 │                     │                     │                    │
     │                 │                     │  _gather_cp_dim()   │                    │
     │                 │                     │  _pool_hidden()     │                    │
     │                 │                     │                     │                    │
     │                 │  embeddings         │                     │                    │
     │                 │◄────────────────────│                     │                    │
     │                 │                     │                     │                    │
     │  Output files   │                     │                     │                    │
     │◄────────────────│                     │                     │                    │
     │                 │                     │                     │                    │
```

## 4. Checkpoint Loading Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT LOADING FOR PARTIAL MODELS                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Full Checkpoint Structure (evo2-7b, 32 layers):
───────────────────────────────────────────────

checkpoint/weights/__0_0.distcp/
├── embedding.word_embeddings.weight           ✓ Load
├── embedding.position_embeddings.weight       ✓ Load (if exists)
├── decoder.layers.0.norm.weight               ✓ Load
├── decoder.layers.0.mixer.*                   ✓ Load
├── decoder.layers.0.mlp.*                     ✓ Load
├── decoder.layers.1.*                         ✓ Load
├── ...
├── decoder.layers.15.*                        ✓ Load
├── decoder.layers.16.*                        ✗ NOT LOADED (layer doesn't exist)
├── ...
├── decoder.layers.31.*                        ✗ NOT LOADED
├── decoder.final_norm.weight                  ✓ Load
└── output_layer.weight                        ✗ NOT LOADED (not needed for embeddings)


Model Configuration for embedding_layer=16:
───────────────────────────────────────────

config = HYENA_MODEL_OPTIONS["7b_arc_longcontext"](
    num_layers=16,                    # Only create 16 layers
    forward_step_fn=embedding_forward_step,
    data_step_fn=hyena_predict_data_step,
)

What happens at load time:
─────────────────────────

1. Model is instantiated with num_layers=16
   → self.decoder.layers has 16 HyenaLayer modules (indices 0-15)
   → output_layer is created but we skip it in forward

2. nl.AutoResume loads checkpoint with ckpt_load_strictness="log_all"
   → Logs warnings for missing keys (layers 16-31, output_layer)
   → Does NOT fail on missing keys
   → Only loads weights that match existing model structure

3. Result: Model with first 16 layers loaded, rest ignored
```

## 5. Forward Pass Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD PREDICTION vs EMBEDDING EXTRACTION                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘

STANDARD PREDICTION (predict.py - existing):
────────────────────────────────────────────

Input: tokens [B, S]
           │
           ▼
    ┌──────────────┐
    │  embedding   │ → [S, B, H]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   decoder    │ → [S, B, H]  (all 32 layers)
    │  (32 layers) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ output_layer │ → [S, B, V]  (V = vocab_size = 512)
    │   (LM head)  │
    └──────┬───────┘
           │
           ▼
    transpose → [B, S, V]
           │
           ▼
    ┌──────────────┐
    │  log_softmax │ → log_probs [B, S, V]
    │   (optional) │
    └──────────────┘

Output: logits [B, S, 512] or log_probs [B]


EMBEDDING EXTRACTION (embed.py - new):
──────────────────────────────────────

Input: tokens [B, S]
           │
           ▼
    ┌──────────────┐
    │  embedding   │ → [S, B, H]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   decoder    │ → [S, B, H]  (only N layers if truncated)
    │  (N layers)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  final_norm  │ → [S, B, H]  (optional, usually yes)
    └──────┬───────┘
           │
           ▼
    ╳ SKIP output_layer
           │
           ▼
    transpose → [B, S, H]
           │
           ▼
    ┌──────────────┐
    │   pooling    │ → [B, H] (for mean/max/first/last)
    │              │   [B, S, H] (for per_token)
    └──────────────┘

Output: embeddings [B, H] where H = hidden_size (4096 for 7B)
```

## 6. Implementation Skeleton

```python
# File: sub-packages/bionemo-evo2/src/bionemo/evo2/models/embedder.py

"""Embedding extraction for Evo2 models."""

from typing import Literal, Optional

import torch
from megatron.core import parallel_state
from torch import Tensor

from bionemo.evo2.run.predict import (
    BasePredictor,
    _gather_along_cp_dim,
    hyena_predict_data_step,
)
from nemo.collections.llm.gpt.model.hyena import HyenaModel
from bionemo.evo2.models.mamba import MambaModel
from nemo.collections.llm.gpt.model.base import GPTModel


PoolingStrategy = Literal["mean", "max", "last", "first", "per_token"]


class EmbeddingExtractorMixin(BasePredictor):
    """Mixin providing embedding extraction capabilities.

    This mixin overrides predict_step() to return hidden state embeddings
    instead of logits. It must be combined with a model class (HyenaModel,
    MambaModel, or GPTModel).

    Attributes:
        embedding_layer: Which layer to extract embeddings from.
            If None, uses all layers. If 0, uses embedding layer output.
        pooling_strategy: How to pool sequence-level embeddings.
        include_final_norm: Whether to apply final layer norm.
    """

    def __init__(
        self,
        *args,
        embedding_layer: Optional[int] = None,
        pooling_strategy: PoolingStrategy = "mean",
        include_final_norm: bool = True,
        **kwargs,
    ):
        # Remove embedding-specific args before passing to parent
        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.pooling_strategy = pooling_strategy
        self.include_final_norm = include_final_norm
        self._cp_unshuffle_warning_shown = False

    def predict_step(
        self, batch, batch_idx: Optional[int] = None, to_cpu: bool = True
    ) -> Optional[dict[str, Tensor]]:
        """Extract embeddings from input batch.

        Args:
            batch: Dictionary with tokens, position_ids, loss_mask, seq_idx
            batch_idx: Batch index (unused)
            to_cpu: Whether to move results to CPU

        Returns:
            Dictionary with:
            - embeddings: [B, H] or [B, S, H] depending on pooling
            - seq_idx: Sequence indices mapping to original FASTA
            - pad_mask: (only for per_token) Valid token mask
        """
        if len(batch) == 0:
            return None

        assert not self.training, "predict_step should be called in eval mode"

        with torch.no_grad():
            hidden_states = self.forward_for_embeddings(batch)

        if not parallel_state.is_pipeline_last_stage():
            return None

        # Gather across Context Parallel ranks
        hidden_gathered = _gather_along_cp_dim(hidden_states, seq_dim=1)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])

        # Handle zigzag for position-sensitive pooling
        cp_size = parallel_state.get_context_parallel_world_size()
        if self.pooling_strategy in ("last", "per_token") and cp_size > 1:
            if not self._cp_unshuffle_warning_shown:
                # Warn user about zigzag ordering
                logger.warning(
                    "Using position-sensitive pooling with CP > 1. "
                    "Results are in zigzag order. Use _unshuffle_zigzag() "
                    "to restore original sequence order if needed."
                )
                self._cp_unshuffle_warning_shown = True

        # Apply pooling
        embeddings = self._pool_hidden_states(
            hidden_gathered,
            loss_mask_gathered,
            self.pooling_strategy,
        )

        result = {
            "embeddings": embeddings.cpu() if to_cpu else embeddings,
            "seq_idx": batch["seq_idx"].cpu() if to_cpu else batch["seq_idx"],
        }

        if self.pooling_strategy == "per_token":
            mask = loss_mask_gathered
            result["pad_mask"] = mask.cpu() if to_cpu else mask

        return result

    def forward_for_embeddings(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass returning hidden states instead of logits.

        This method runs the model up to (but not including) the output layer,
        returning the decoder's hidden states.

        Args:
            batch: Input batch with tokens and position_ids

        Returns:
            hidden_states: [B, S, H] tensor of hidden states
        """
        # Get the underlying Megatron model
        # In NeMo, self.module is the MCoreHyenaModel
        model = self.module

        input_ids = batch["tokens"]
        position_ids = batch["position_ids"]

        # Step 1: Run embedding layer
        if model.pre_process:
            decoder_input = model.embedding(
                input_ids=input_ids,
                position_ids=position_ids
            )
        else:
            # For pipeline parallel intermediate stages
            decoder_input = None

        # Step 2: Get rotary embeddings if needed
        rotary_pos_emb = None
        if hasattr(model, 'rotary_pos_emb') and model.rotary_pos_emb is not None:
            rotary_pos_emb = model.rotary_pos_emb(
                model.max_sequence_length
            )

        # Step 3: Run decoder (truncated if num_layers was set in config)
        hidden_states = model.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Step 4: Apply final norm if requested
        if (self.include_final_norm and
            hasattr(model.decoder, 'final_norm') and
            model.decoder.final_norm is not None):
            hidden_states = model.decoder.final_norm(hidden_states)

        # Step 5: Transpose from [S, B, H] to [B, S, H]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        return hidden_states

    def _pool_hidden_states(
        self,
        hidden_states: Tensor,
        mask: Tensor,
        strategy: PoolingStrategy,
    ) -> Tensor:
        """Apply pooling strategy to hidden states.

        Args:
            hidden_states: [B, S, H] hidden states
            mask: [B, S] boolean mask (True = valid token)
            strategy: Pooling strategy name

        Returns:
            Pooled embeddings: [B, H] for most strategies,
            [B, S, H] for per_token
        """
        mask_float = mask.float().unsqueeze(-1)  # [B, S, 1]

        if strategy == "per_token":
            return hidden_states * mask_float

        elif strategy == "mean":
            masked_sum = (hidden_states * mask_float).sum(dim=1)
            valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
            return masked_sum / valid_counts

        elif strategy == "max":
            masked = hidden_states.masked_fill(
                ~mask.unsqueeze(-1).bool(),
                float('-inf')
            )
            return masked.max(dim=1).values

        elif strategy == "last":
            seq_lengths = mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(
                hidden_states.size(0),
                device=hidden_states.device
            )
            return hidden_states[batch_idx, seq_lengths]

        elif strategy == "first":
            return hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")


class HyenaEmbedder(EmbeddingExtractorMixin, HyenaModel):
    """Hyena model for embedding extraction.

    Combines EmbeddingExtractorMixin with HyenaModel to provide
    embedding extraction capabilities for Evo2 Hyena models.
    """

    def configure_model(self, *args, **kwargs) -> None:
        """Configure the model."""
        super().configure_model(*args, **kwargs)
        self.trainer.strategy._init_model_parallel = True


class MambaEmbedder(EmbeddingExtractorMixin, MambaModel):
    """Mamba model for embedding extraction."""
    pass


class LlamaEmbedder(EmbeddingExtractorMixin, GPTModel):
    """Llama/Eden model for embedding extraction."""
    pass


def embedding_forward_step(model, batch) -> Tensor:
    """Forward step that returns hidden states for embedding.

    This is the forward_step_fn to be passed to model config.
    Unlike hyena_predict_forward_step, this calls forward_for_embeddings.
    """
    return model.forward_for_embeddings(batch)


def _unshuffle_zigzag(
    tensor: Tensor,
    cp_size: int,
    seq_dim: int = 1
) -> Tensor:
    """Restore original sequence order from zigzag-packed tensor.

    After Context Parallel gather, sequences are in zigzag order.
    This function restores the original sequential order.

    Args:
        tensor: Tensor with zigzag-ordered sequence dimension
        cp_size: Context parallel world size
        seq_dim: Which dimension contains the sequence

    Returns:
        Tensor with original sequence ordering
    """
    if cp_size == 1:
        return tensor

    num_chunks = 2 * cp_size
    chunks = tensor.chunk(num_chunks, dim=seq_dim)

    # Reconstruct original order from zigzag pattern
    original_order = [None] * num_chunks
    chunk_idx = 0
    for rank in range(cp_size):
        original_order[rank] = chunks[chunk_idx]
        chunk_idx += 1
        original_order[num_chunks - 1 - rank] = chunks[chunk_idx]
        chunk_idx += 1

    return torch.cat(original_order, dim=seq_dim)
```

## 7. CLI Entry Point Skeleton

```python
# File: sub-packages/bionemo-evo2/src/bionemo/evo2/run/embed.py

"""CLI for Evo2 embedding extraction."""

import argparse
from pathlib import Path
from typing import Literal

# ... imports similar to predict.py ...

from bionemo.evo2.models.embedder import (
    HyenaEmbedder,
    MambaEmbedder,
    LlamaEmbedder,
    embedding_forward_step,
)


def parse_args():
    """Parse arguments for embedding extraction."""
    ap = argparse.ArgumentParser(
        description="Extract embeddings from Evo2 models"
    )

    # Reuse most args from predict.py
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-size", type=str, default="7b_arc_longcontext")

    # Parallelism args (same as predict.py)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--context-parallel-size", type=int, default=1)
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--num-nodes", type=int, default=1)

    # Embedding-specific args (NEW)
    ap.add_argument(
        "--embedding-layer",
        type=int,
        default=None,
        help="Layer to extract from (None=all layers, N=after layer N)"
    )
    ap.add_argument(
        "--pooling-strategy",
        type=str,
        choices=["mean", "max", "last", "first", "per_token"],
        default="mean",
    )
    ap.add_argument(
        "--no-final-norm",
        action="store_true",
        help="Skip final layer normalization"
    )

    # Other args from predict.py
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--min-length", type=int, default=None)
    ap.add_argument("--fp8", action="store_true")
    # ... etc

    return ap.parse_args()


def extract_embeddings(
    fasta_path: Path,
    ckpt_dir: Path,
    output_dir: Path,
    embedding_layer: int | None = None,
    pooling_strategy: str = "mean",
    # ... other params from predict() ...
):
    """Main embedding extraction function.

    This function is largely similar to predict() but:
    1. Uses HyenaEmbedder instead of HyenaPredictor
    2. Passes embedding_layer and pooling_strategy to model
    3. Uses embedding_forward_step as forward_step_fn
    """

    # Determine effective num_layers from embedding_layer
    if embedding_layer is not None:
        config_num_layers = embedding_layer
    else:
        config_num_layers = None  # Use default from model config

    # Build config with num_layers override
    config = HYENA_MODEL_OPTIONS[model_size](
        forward_step_fn=embedding_forward_step,
        data_step_fn=hyena_predict_data_step,
        num_layers=config_num_layers,  # Key: truncate model
        # ... other config ...
    )

    # Create embedder model
    model = HyenaEmbedder(
        config,
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        pooling_strategy=pooling_strategy,
        include_final_norm=not no_final_norm,
    )

    # Rest is similar to predict():
    # - Create trainer
    # - Load checkpoint with AutoResume
    # - Create datamodule
    # - Run trainer.predict()

    # ...


def main():
    args = parse_args()
    extract_embeddings(
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        # ... map all args ...
    )


if __name__ == "__main__":
    main()
```

## 8. Testing Strategy

```python
# File: tests/bionemo/evo2/test_embedder.py

"""Tests for embedding extraction."""

import pytest
import torch

from bionemo.evo2.models.embedder import (
    EmbeddingExtractorMixin,
    _pool_hidden_states,
    _unshuffle_zigzag,
)


class TestPoolingStrategies:
    """Test pooling operations."""

    @pytest.fixture
    def sample_hidden_states(self):
        """Create sample hidden states [B=2, S=4, H=8]."""
        return torch.randn(2, 4, 8)

    @pytest.fixture
    def sample_mask(self):
        """Mask with varying sequence lengths."""
        return torch.tensor([
            [True, True, True, False],  # length 3
            [True, True, False, False],  # length 2
        ])

    def test_mean_pooling(self, sample_hidden_states, sample_mask):
        """Mean pooling respects mask."""
        result = _pool_hidden_states(
            sample_hidden_states, sample_mask, "mean"
        )
        assert result.shape == (2, 8)
        # Verify mean is computed only over valid tokens

    def test_max_pooling(self, sample_hidden_states, sample_mask):
        """Max pooling ignores padded positions."""
        result = _pool_hidden_states(
            sample_hidden_states, sample_mask, "max"
        )
        assert result.shape == (2, 8)

    def test_last_pooling(self, sample_hidden_states, sample_mask):
        """Last pooling gets correct token."""
        result = _pool_hidden_states(
            sample_hidden_states, sample_mask, "last"
        )
        assert result.shape == (2, 8)
        # First batch should get position 2, second should get position 1

    def test_first_pooling(self, sample_hidden_states, sample_mask):
        """First pooling always gets position 0."""
        result = _pool_hidden_states(
            sample_hidden_states, sample_mask, "first"
        )
        assert result.shape == (2, 8)
        torch.testing.assert_close(
            result, sample_hidden_states[:, 0, :]
        )

    def test_per_token(self, sample_hidden_states, sample_mask):
        """Per-token returns full sequence."""
        result = _pool_hidden_states(
            sample_hidden_states, sample_mask, "per_token"
        )
        assert result.shape == (2, 4, 8)


class TestZigzagUnshuffle:
    """Test zigzag unshuffling for CP."""

    def test_unshuffle_cp2(self):
        """Test unshuffling with CP=2."""
        # Simulated zigzag order: [0,1,6,7,2,3,4,5]
        tensor = torch.tensor([0,1,6,7,2,3,4,5]).unsqueeze(0).unsqueeze(-1)
        result = _unshuffle_zigzag(tensor, cp_size=2, seq_dim=1)
        expected = torch.tensor([0,1,2,3,4,5,6,7]).unsqueeze(0).unsqueeze(-1)
        torch.testing.assert_close(result, expected)

    def test_unshuffle_cp1_noop(self):
        """CP=1 should return input unchanged."""
        tensor = torch.randn(2, 8, 4)
        result = _unshuffle_zigzag(tensor, cp_size=1)
        torch.testing.assert_close(result, tensor)
```

---

## 9. Quick Reference Card

| Component | File | Purpose |
|-----------|------|---------|
| `EmbeddingExtractorMixin` | `models/embedder.py` | Embedding extraction logic |
| `HyenaEmbedder` | `models/embedder.py` | Hyena + embedding mixin |
| `MambaEmbedder` | `models/embedder.py` | Mamba + embedding mixin |
| `LlamaEmbedder` | `models/embedder.py` | Llama + embedding mixin |
| `embedding_forward_step` | `models/embedder.py` | Forward step function for embeddings |
| `_unshuffle_zigzag` | `models/embedder.py` | CP zigzag ordering fix |
| `extract_embeddings` | `run/embed.py` | Main entry point function |
| `parse_args` | `run/embed.py` | CLI argument parsing |

| Reused Component | Source | Purpose |
|------------------|--------|---------|
| `SimpleFastaDataset` | `data/fasta_dataset.py` | FASTA loading/tokenization |
| `PredictDataModule` | `run/predict.py` | Batching with padding |
| `hyena_predict_data_step` | `run/predict.py` | Data step with CP slicing |
| `_gather_along_cp_dim` | `run/predict.py` | CP tensor gathering |
| `PredictionWriter` | `bionemo.llm.utils.callbacks` | Output file writing |
| `HYENA_MODEL_OPTIONS` | `nemo.collections.llm` | Model configurations |
