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
from nemo.utils import logging as logger
from torch import Tensor

from bionemo.evo2.models.mamba import MambaModel
from bionemo.evo2.run.predict import _gather_along_cp_dim

# Import BasePredictor and helper from predict.py
# Assuming BasePredictor is just LightningPassthroughPredictionMixin as per docs,
# but in predict.py HyenaPredictor inherits HyenaModel.
# Wait, predict.py doesn't define HyenaPredictor explicitly, it just instantiates HyenaModel.
# But HyenaModel inherits LightningPassthroughPredictionMixin?
# Let's check HyenaModel definition in NeMo if possible, but we can't.
# However, `predict.py` imports `LightningPassthroughPredictionMixin` from `bionemo.llm.lightning`.
# The design doc says:
# class HyenaEmbedder(EmbeddingExtractorMixin, HyenaModel): pass
# And EmbeddingExtractorMixin(BasePredictor).
# "BasePredictor" is marked as [existing] in diagram and mapped to LightningPassthroughPredictionMixin in text.
# So I will use LightningPassthroughPredictionMixin as the base if there is no explicit BasePredictor class.
from bionemo.llm.lightning import LightningPassthroughPredictionMixin


PoolingStrategy = Literal["mean", "max", "last", "first", "per_token"]


class EmbeddingExtractorMixin(LightningPassthroughPredictionMixin):
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
        # We need to filter kwargs commonly passed to the model constructor if they are not expected by parent
        # But here we are mixing in. The model __init__ will likely consume these if we don't handle them?
        # No, usually mixin __init__ is called, removes its args, then calls super().__init__.
        # But since we come *before* the Model in MRO, super() is the Model!
        # So we should pass remaining kwargs to super().__init__.

        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.pooling_strategy = pooling_strategy
        self.include_final_norm = include_final_norm
        self._cp_unshuffle_warning_shown = False

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
        if self.training:
            self.eval()

        with torch.no_grad():
            hidden_states = self.forward_for_embeddings(batch)

        if not parallel_state.is_pipeline_last_stage():
            return None

        # Gather across Context Parallel ranks
        # hidden_states is [B, S, H]. gather along seq_dim=1.
        hidden_gathered = _gather_along_cp_dim(hidden_states, seq_dim=1)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])

        # We also need to gather tokens if we were returning them, but we aren't.
        # But we need loss_mask for pooling.

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
        """
        mask_float = mask.float().unsqueeze(-1)  # [B, S, 1]

        if strategy == "per_token":
            return hidden_states * mask_float

        elif strategy == "mean":
            masked_sum = (hidden_states * mask_float).sum(dim=1)
            valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
            return masked_sum / valid_counts

        elif strategy == "max":
            # Mask invalid positions with -inf
            masked = hidden_states.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            # max returns (values, indices)
            return masked.max(dim=1).values

        elif strategy == "last":
            # Find the index of the last valid token
            seq_lengths = mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]

        elif strategy == "first":
            return hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")


class HyenaEmbedder(EmbeddingExtractorMixin, HyenaModel):
    """Hyena model with embedding extraction capabilities."""

    pass


class MambaEmbedder(EmbeddingExtractorMixin, MambaModel):
    """Mamba model with embedding extraction capabilities."""

    pass


class LlamaEmbedder(EmbeddingExtractorMixin, GPTModel):
    """Llama model with embedding extraction capabilities."""

    pass
