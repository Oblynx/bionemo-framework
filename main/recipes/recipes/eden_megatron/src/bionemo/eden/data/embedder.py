# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding extraction for Evo2 models."""

from typing import Any, Dict, Literal, Optional

import torch
from megatron.core import parallel_state
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.hyena import HyenaModel
from torch import Tensor

from bionemo.evo2.models.mamba import MambaModel
from bionemo.evo2.run.predict import BasePredictor, _gather_along_cp_dim


PoolingStrategy = Literal["mean", "max", "last", "first", "per_token"]


def _unshuffle_zigzag(tensor: Tensor, cp_size: int, seq_dim: int = 1) -> Tensor:
    """Restore original sequence order from zigzag-packed tensor.

    After Context Parallel gather, sequences are in zigzag order:
    [chunk_0, chunk_{2*cp_size-1}, chunk_1, chunk_{2*cp_size-2}, ...]

    This function restores the original sequential order:
    [chunk_0, chunk_1, chunk_2, ..., chunk_{2*cp_size-1}]

    Args:
        tensor: Tensor with zigzag-ordered sequence dimension
        cp_size: Context parallel world size
        seq_dim: Which dimension contains the sequence (default: 1)

    Returns:
        Tensor with original sequence ordering

    Examples:
        >>> # With CP=2, input has 4 chunks in zigzag order: [0,3,1,2]
        >>> # Output should be: [0,1,2,3]
        >>> tensor = torch.tensor([[0,0,3,3,1,1,2,2]])  # [B, S]
        >>> result = _unshuffle_zigzag(tensor, cp_size=2, seq_dim=1)
        >>> # result: [[0,0,1,1,2,2,3,3]]
    """
    if cp_size == 1:
        return tensor

    num_chunks = 2 * cp_size
    chunks = list(tensor.chunk(num_chunks, dim=seq_dim))

    # Reconstruct original order from zigzag pattern
    # Zigzag pattern: rank r gets chunks [r*2, num_chunks - 1 - r*2]
    original_order = [None] * num_chunks
    chunk_idx = 0
    for rank in range(cp_size):
        original_order[rank * 2] = chunks[chunk_idx]
        chunk_idx += 1
        original_order[num_chunks - 1 - rank * 2] = chunks[chunk_idx]
        chunk_idx += 1

    return torch.cat(original_order, dim=seq_dim)


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
        """Initialize embedding extraction mixin.

        Args:
            *args: Additional positional arguments passed to parent class.
            embedding_layer: Which layer to extract embeddings from. None for all layers, 0 for embedding layer only.
            pooling_strategy: How to pool sequence-level embeddings.
            include_final_norm: Whether to apply final layer norm.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        # Remove embedding-specific args before passing to parent
        # We need to filter kwargs commonly passed to the model constructor if they are not expected by parent
        # But here we are mixing in. The model __init__ will likely consume these if we don't handle them?
        # No, usually mixin __init__ is called, removes its args, then calls super().__init__.
        # But since we come *before* the Model in MRO, super() is the Model!
        # So we should pass remaining kwargs to super().__init__.

        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.pooling_strategy = pooling_strategy
        self.include_final_norm = include_final_norm

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: Optional[int] = None, to_cpu: bool = True
    ) -> Optional[Dict[str, Any]]:
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
        # Note: batch could be empty or None? predict.py data step handles yielding.
        if batch is None or len(batch) == 0:
            return None

        # Ensure we are in eval mode
        assert not self.training, "predict_step should be called in eval mode. Call model.eval() before predict."

        with torch.no_grad():
            hidden_states = self.forward_for_embeddings(batch)

        # Gather across TP ranks if hidden dimension is sharded
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            from megatron.core.tensor_parallel.mappings import _gather_along_last_dim

            hidden_states = _gather_along_last_dim(
                hidden_states, group=parallel_state.get_tensor_model_parallel_group()
            )

        if not parallel_state.is_pipeline_last_stage():
            return None

        # Gather across Context Parallel ranks
        # hidden_states is [B, S, H]. gather along seq_dim=1.
        hidden_gathered = _gather_along_cp_dim(hidden_states, seq_dim=1)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])

        # Unshuffle zigzag ordering for position-sensitive pooling
        cp_size = parallel_state.get_context_parallel_world_size()
        if self.pooling_strategy in ("last", "per_token") and cp_size > 1:
            hidden_gathered = _unshuffle_zigzag(hidden_gathered, cp_size, seq_dim=1)
            loss_mask_gathered = _unshuffle_zigzag(loss_mask_gathered, cp_size, seq_dim=1)

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

    def forward_for_embeddings(self, batch: Dict[str, Tensor]) -> Tensor:
        """Forward pass returning hidden states instead of logits.

        This method runs the model up to (but not including) the output layer,
        returning the decoder's hidden states.

        Args:
            batch: Input batch with tokens and position_ids

        Returns:
            hidden_states: [B, S, H] tensor of hidden states
        """
        # Get the underlying Megatron model
        # For NeMo models inheriting from GPTModel, self.module is usually the wrapped Megatron module.
        # We access it via self.module if it exists, otherwise self.
        model = getattr(self, "module", self)

        # In some versions it might be self.model... check predict.py usage
        # predict.py: hyena_predict_forward_step(model, batch) -> model(**forward_args)
        # So calling the model instance directly runs forward.
        # But here we want custom forward path.
        # The design doc says "model = self.module".

        input_ids = batch["tokens"]
        position_ids = batch["position_ids"]

        # Step 1: Run embedding layer
        # Check if pre_process is True (first stage logic)
        if hasattr(model, "pre_process") and model.pre_process:
            # Calling embedding layer.
            # MCoreHyenaModel/GPTModel typically has `embedding` attribute which is `LanguageModelEmbedding`
            # helper or similar.
            # Design doc says: model.embedding(input_ids=input_ids, position_ids=position_ids)
            # Let's trust design doc.
            decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # For pipeline parallel intermediate stages
            decoder_input = None

        # Step 2: Get rotary embeddings if needed
        rotary_pos_emb = None
        if hasattr(model, "rotary_pos_emb") and model.rotary_pos_emb is not None:
            rotary_pos_emb = model.rotary_pos_emb(model.max_sequence_length)

        # Early exit for embedding layer (layer 0)
        if self.embedding_layer == 0:
            if decoder_input is None:
                # Should not happen on first stage, but if pipeline parallel?
                # For inference PP=1 usually.
                raise RuntimeError("decoder_input is None but requesting embedding_layer=0")

            # Transpose to [B, S, H]
            return decoder_input.transpose(0, 1).contiguous()

        # Step 3: Run decoder (truncated if num_layers was set in config)
        # model.decoder is typically a TransformerBlock / HyenaDecoder
        hidden_states = model.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Step 4: Apply final norm if requested
        if (
            self.include_final_norm
            and
            # 'post_process' check usually means we are at the last stage
            (not hasattr(model, "post_process") or model.post_process)
            and hasattr(model.decoder, "final_norm")
            and model.decoder.final_norm is not None
        ):
            hidden_states = model.decoder.final_norm(hidden_states)

        # Step 5: Transpose from [S, B, H] to [B, S, H]
        # Megatron usually keeps [S, B, H]. We want [B, S, H] for pooling.
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

        Raises:
            ValueError: If any sequence has no valid tokens and strategy
                requires valid tokens (mean, max, last)
        """
        # Validate that sequences have valid tokens (except for per_token)
        if strategy != "per_token":
            batch_has_valid = mask.any(dim=1)
            if not batch_has_valid.all():
                invalid_indices = torch.where(~batch_has_valid)[0].tolist()
                raise ValueError(
                    f"Cannot apply '{strategy}' pooling to sequences with no valid tokens. "
                    f"Batch items with all-False masks: {invalid_indices}. "
                    f"Check your input sequences for empty/all-padding data."
                )

        mask_float = mask.float().unsqueeze(-1)  # [B, S, 1]

        if strategy == "per_token":
            return hidden_states * mask_float

        elif strategy == "mean":
            masked_sum = (hidden_states * mask_float).sum(dim=1)
            valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
            return masked_sum / valid_counts

        elif strategy == "max":
            # Validation already done above
            masked = hidden_states.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            return masked.max(dim=1).values

        elif strategy == "last":
            # Validation already done above
            seq_lengths = mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
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
    """Mamba model with embedding extraction capabilities."""

    pass


class LlamaEmbedder(EmbeddingExtractorMixin, GPTModel):
    """Llama model with embedding extraction capabilities."""

    pass
