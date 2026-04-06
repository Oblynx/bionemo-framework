"""Tests for Evo2 embedding extraction functionality.

These tests are designed to run on CPU with mock models for fast CI/CD execution.
No GPU or large checkpoints are required.
"""

from typing import Dict, Optional
from unittest import mock

import pytest
import torch
from torch import Tensor, nn

from bionemo.evo2.models.embedder import (
    EmbeddingExtractorMixin,
    PoolingStrategy,
    _pool_hidden_states,
    _unshuffle_zigzag,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_hidden_states() -> Tensor:
    """Create sample hidden states [B=3, S=4, H=8]."""
    torch.manual_seed(42)
    return torch.randn(3, 4, 8)


@pytest.fixture
def sample_mask() -> Tensor:
    """Mask with varying sequence lengths.

    Batch 0: 3 valid tokens
    Batch 1: 2 valid tokens
    Batch 2: 4 valid tokens (all)
    """
    return torch.tensor(
        [
            [True, True, True, False],  # length 3
            [True, True, False, False],  # length 2
            [True, True, True, True],  # length 4
        ]
    )


@pytest.fixture
def empty_mask() -> Tensor:
    """Mask with some sequences having no valid tokens."""
    return torch.tensor(
        [
            [False, False, False, False],  # all invalid
            [True, True, False, False],  # valid
            [False, False, False, False],  # all invalid
        ]
    )


# =============================================================================
# Unit Tests: _unshuffle_zigzag
# =============================================================================


class TestUnshuffleZigzag:
    """Unit tests for the _unshuffle_zigzag function."""

    def test_cp1_noop(self):
        """CP=1 should return input unchanged."""
        tensor = torch.randn(2, 8, 4)  # [B, S, H]
        result = _unshuffle_zigzag(tensor, cp_size=1, seq_dim=1)
        torch.testing.assert_close(result, tensor)

    def test_cp2_unshuffling(self):
        """Test unshuffling with CP=2 produces correct sequential order.

        With CP=2, sequence is split into 4 chunks in zigzag order:
        - Rank 0 gets: [chunk_0, chunk_3]
        - Rank 1 gets: [chunk_1, chunk_2]
        - After gather: [chunk_0, chunk_3, chunk_1, chunk_2]
        - Should restore to: [chunk_0, chunk_1, chunk_2, chunk_3]
        """
        # Create tensor with zigzag order: [0, 0, 3, 3, 1, 1, 2, 2]
        # Each chunk is 2 elements
        zigzag_tensor = torch.tensor([[0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0]])  # [1, 8]

        result = _unshuffle_zigzag(zigzag_tensor, cp_size=2, seq_dim=1)

        # Expected: [0, 0, 1, 1, 2, 2, 3, 3]
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
        torch.testing.assert_close(result, expected)

    def test_cp4_unshuffling(self):
        """Test unshuffling with CP=4.

        With CP=4, 8 chunks in zigzag order:
        - Ranks get: [0,7], [1,6], [2,5], [3,4]
        - After gather: [0,7,1,6,2,5,3,4]
        - Should restore to: [0,1,2,3,4,5,6,7]
        """
        # Zigzag order: [0,7,1,6,2,5,3,4]
        zigzag_indices = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])
        zigzag_tensor = torch.arange(8).float()[zigzag_indices].unsqueeze(0)  # [1, 8]

        result = _unshuffle_zigzag(zigzag_tensor, cp_size=4, seq_dim=1)

        # Expected: [0,1,2,3,4,5,6,7]
        expected = torch.arange(8).float().unsqueeze(0)
        torch.testing.assert_close(result, expected)

    def test_3d_tensor(self):
        """Test unshuffling works correctly on [B, S, H] tensors."""
        batch_size = 2
        hidden_size = 4
        seq_len = 8

        # Create 3D tensor with sequential values
        sequential = torch.arange(batch_size * seq_len * hidden_size).float()
        sequential = sequential.reshape(batch_size, seq_len, hidden_size)

        # Manually create zigzag ordering for CP=2: chunks [0,3,1,2]
        # Each chunk is 2 positions
        zigzag = torch.cat(
            [
                sequential[:, 0:2, :],  # chunk 0
                sequential[:, 6:8, :],  # chunk 3
                sequential[:, 2:4, :],  # chunk 1
                sequential[:, 4:6, :],  # chunk 2
            ],
            dim=1,
        )

        result = _unshuffle_zigzag(zigzag, cp_size=2, seq_dim=1)

        torch.testing.assert_close(result, sequential)

    def test_different_seq_dim(self):
        """Test unshuffling with seq_dim=0."""
        # Shape [S, B] with S=8, B=2
        zigzag_tensor = torch.tensor([[0.0, 0.0], [3.0, 3.0], [1.0, 1.0], [2.0, 2.0]]).repeat(2, 1)  # [8, 2]
        # Reorder to zigzag: positions [0,1, 6,7, 2,3, 4,5] -> [0,0,3,3,1,1,2,2] per batch

        # Simpler test: just verify shape is preserved and no error
        result = _unshuffle_zigzag(zigzag_tensor, cp_size=2, seq_dim=0)
        assert result.shape == zigzag_tensor.shape


# =============================================================================
# Unit Tests: Pooling Strategies
# =============================================================================


class TestPoolingStrategies:
    """Unit tests for the _pool_hidden_states function."""

    def test_mean_pooling(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """Mean pooling should compute mean only over valid tokens."""
        result = _pool_hidden_states(sample_hidden_states, sample_mask, "mean")

        # Check shape
        assert result.shape == (3, 8), f"Expected shape (3, 8), got {result.shape}"

        # Verify computation for first batch item (positions 0, 1, 2)
        expected_batch0 = sample_hidden_states[0, :3, :].mean(dim=0)
        torch.testing.assert_close(result[0], expected_batch0)

        # Verify for second batch item (positions 0, 1)
        expected_batch1 = sample_hidden_states[1, :2, :].mean(dim=0)
        torch.testing.assert_close(result[1], expected_batch1)

        # Verify for third batch item (all positions)
        expected_batch2 = sample_hidden_states[2, :, :].mean(dim=0)
        torch.testing.assert_close(result[2], expected_batch2)

    def test_max_pooling(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """Max pooling should find max over valid tokens only."""
        result = _pool_hidden_states(sample_hidden_states, sample_mask, "max")

        # Check shape
        assert result.shape == (3, 8)

        # Verify computation for first batch item
        expected_batch0 = sample_hidden_states[0, :3, :].max(dim=0).values
        torch.testing.assert_close(result[0], expected_batch0)

        # Check that padding positions don't affect result
        modified_hidden = sample_hidden_states.clone()
        modified_hidden[0, 3, :] = 1e6  # Huge value in padding position

        result_modified = _pool_hidden_states(modified_hidden, sample_mask, "max")
        torch.testing.assert_close(result_modified[0], expected_batch0)

    def test_last_pooling(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """Last pooling should get the last valid token."""
        result = _pool_hidden_states(sample_hidden_states, sample_mask, "last")

        # Check shape
        assert result.shape == (3, 8)

        # Batch 0: last valid is position 2 (3rd token)
        torch.testing.assert_close(result[0], sample_hidden_states[0, 2, :])

        # Batch 1: last valid is position 1 (2nd token)
        torch.testing.assert_close(result[1], sample_hidden_states[1, 1, :])

        # Batch 2: last valid is position 3 (4th token, all valid)
        torch.testing.assert_close(result[2], sample_hidden_states[2, 3, :])

    def test_first_pooling(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """First pooling should always get position 0."""
        result = _pool_hidden_states(sample_hidden_states, sample_mask, "first")

        # Check shape
        assert result.shape == (3, 8)

        # Should be identical to [:, 0, :]
        expected = sample_hidden_states[:, 0, :]
        torch.testing.assert_close(result, expected)

    def test_per_token_pooling(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """Per-token returns full sequence with invalid positions zeroed."""
        result = _pool_hidden_states(sample_hidden_states, sample_mask, "per_token")

        # Check shape (should preserve sequence dimension)
        assert result.shape == (3, 4, 8)

        # Check that valid positions are preserved
        torch.testing.assert_close(result[0, 0, :], sample_hidden_states[0, 0, :])
        torch.testing.assert_close(result[0, 1, :], sample_hidden_states[0, 1, :])
        torch.testing.assert_close(result[0, 2, :], sample_hidden_states[0, 2, :])

        # Check that invalid positions are zeroed
        expected_zero = torch.zeros(8)
        torch.testing.assert_close(result[0, 3, :], expected_zero)  # Batch 0, position 3
        torch.testing.assert_close(result[1, 2, :], expected_zero)  # Batch 1, position 2
        torch.testing.assert_close(result[1, 3, :], expected_zero)  # Batch 1, position 3


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sequence_mean_raises_error(self, sample_hidden_states: Tensor, empty_mask: Tensor):
        """Mean pooling should raise ValueError for sequences with no valid tokens."""
        with pytest.raises(ValueError, match="no valid tokens"):
            _pool_hidden_states(sample_hidden_states, empty_mask, "mean")

    def test_empty_sequence_max_raises_error(self, sample_hidden_states: Tensor, empty_mask: Tensor):
        """Max pooling should raise ValueError for empty sequences."""
        with pytest.raises(ValueError, match="no valid tokens"):
            _pool_hidden_states(sample_hidden_states, empty_mask, "max")

    def test_empty_sequence_last_raises_error(self, sample_hidden_states: Tensor, empty_mask: Tensor):
        """Last pooling should raise ValueError for empty sequences."""
        with pytest.raises(ValueError, match="no valid tokens"):
            _pool_hidden_states(sample_hidden_states, empty_mask, "last")

    def test_empty_sequence_per_token_works(self, sample_hidden_states: Tensor, empty_mask: Tensor):
        """Per-token pooling should work with empty sequences (returns all zeros)."""
        result = _pool_hidden_states(sample_hidden_states, empty_mask, "per_token")

        # Check shape
        assert result.shape == (3, 4, 8)

        # Batch 0 and 2 should be all zeros (no valid tokens)
        torch.testing.assert_close(result[0], torch.zeros(4, 8))
        torch.testing.assert_close(result[2], torch.zeros(4, 8))

        # Batch 1 should have non-zero values at positions 0, 1
        assert not torch.allclose(result[1, 0, :], torch.zeros(8))
        assert not torch.allclose(result[1, 1, :], torch.zeros(8))
        # But positions 2, 3 should be zero
        torch.testing.assert_close(result[1, 2, :], torch.zeros(8))
        torch.testing.assert_close(result[1, 3, :], torch.zeros(8))

    def test_invalid_pooling_strategy(self, sample_hidden_states: Tensor, sample_mask: Tensor):
        """Should raise ValueError for unknown pooling strategy."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            _pool_hidden_states(sample_hidden_states, sample_mask, "invalid_strategy")

    def test_single_token_sequence(self):
        """Test pooling with sequences containing only one valid token."""
        hidden_states = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [
                [True, False, False, False],  # only position 0 valid
                [False, False, True, False],  # only position 2 valid
            ]
        )

        # Mean should equal the single valid token
        mean_result = _pool_hidden_states(hidden_states, mask, "mean")
        torch.testing.assert_close(mean_result[0], hidden_states[0, 0, :])
        torch.testing.assert_close(mean_result[1], hidden_states[1, 2, :])

        # Max should equal the single valid token
        max_result = _pool_hidden_states(hidden_states, mask, "max")
        torch.testing.assert_close(max_result[0], hidden_states[0, 0, :])
        torch.testing.assert_close(max_result[1], hidden_states[1, 2, :])

        # Last should equal the single valid token
        last_result = _pool_hidden_states(hidden_states, mask, "last")
        torch.testing.assert_close(last_result[0], hidden_states[0, 0, :])
        torch.testing.assert_close(last_result[1], hidden_states[1, 2, :])

    def test_all_tokens_valid(self):
        """Test pooling when all tokens in all sequences are valid."""
        hidden_states = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)

        # All strategies should work
        for strategy in ["mean", "max", "last", "first", "per_token"]:
            result = _pool_hidden_states(hidden_states, mask, strategy)
            if strategy == "per_token":
                assert result.shape == (2, 4, 8)
            else:
                assert result.shape == (2, 8)


# =============================================================================
# Mock Model for Integration Tests
# =============================================================================


class MockDecoder(nn.Module):
    """Mock decoder that mimics Megatron decoder interface."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through decoder layers.

        Args:
            hidden_states: Input tensor [S, B, H]
            attention_mask: Unused, for interface compatibility
            rotary_pos_emb: Unused, for interface compatibility

        Returns:
            Output tensor [S, B, H]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MockMegatronModel(nn.Module):
    """Minimal mock of Megatron model for testing the embedder.

    This model implements the minimal interface required by EmbeddingExtractorMixin:
    - embedding(input_ids, position_ids) -> [S, B, H]
    - decoder(hidden_states, ...) -> [S, B, H]
    - decoder.final_norm
    - pre_process, post_process attributes
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, vocab_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.pre_process = True
        self.post_process = True
        self.max_sequence_length = 1024
        self.rotary_pos_emb = None

        # Mock embedding
        self._embedding = nn.Embedding(vocab_size, hidden_size)

        # Mock decoder
        self.decoder = MockDecoder(hidden_size, num_layers)

    def embedding(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Compute embeddings from input IDs.

        Args:
            input_ids: Token IDs [B, S]
            position_ids: Position IDs (unused in mock)

        Returns:
            Embeddings [S, B, H] (Megatron format)
        """
        # [B, S] -> [B, S, H]
        embedded = self._embedding(input_ids)
        # Transpose to Megatron format [S, B, H]
        return embedded.transpose(0, 1).contiguous()


class MockEmbedder(EmbeddingExtractorMixin):
    """Mock embedder using MockMegatronModel for testing.

    This class combines EmbeddingExtractorMixin with our mock model
    to test the full predict_step flow without requiring the full
    Megatron/NeMo infrastructure.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        vocab_size: int = 512,
        embedding_layer: Optional[int] = None,
        pooling_strategy: PoolingStrategy = "mean",
        include_final_norm: bool = True,
    ):
        # Skip parent __init__ since we're not using real NeMo model
        # Just set the attributes directly
        self.embedding_layer = embedding_layer
        self.pooling_strategy = pooling_strategy
        self.include_final_norm = include_final_norm
        self.training = False

        # Create mock model
        self.module = MockMegatronModel(hidden_size, num_layers, vocab_size)

    def eval(self):
        """Set to eval mode."""
        self.training = False
        self.module.eval()
        return self


# =============================================================================
# Integration Tests with Mock Model
# =============================================================================


class TestIntegration:
    """Integration tests using MockEmbedder."""

    @pytest.fixture
    def mock_parallel_state(self):
        """Mock parallel state for single-rank testing (CP=1, TP=1, PP=1)."""
        with (
            mock.patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=1),
            mock.patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=1),
            mock.patch("megatron.core.parallel_state.is_pipeline_last_stage", return_value=True),
            mock.patch("bionemo.evo2.run.predict._gather_along_cp_dim", side_effect=lambda x, **kwargs: x),
        ):
            yield

    @pytest.fixture
    def sample_batch(self) -> Dict[str, Tensor]:
        """Create a sample batch for testing."""
        batch_size = 2
        seq_len = 8
        return {
            "tokens": torch.randint(0, 512, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "loss_mask": torch.tensor(
                [
                    [True, True, True, True, False, False, False, False],
                    [True, True, True, True, True, True, False, False],
                ]
            ),
            "seq_idx": torch.tensor([0, 1]),
        }

    def test_predict_step_mean_pooling(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test full predict_step with mean pooling."""
        embedder = MockEmbedder(
            hidden_size=64,
            num_layers=2,
            pooling_strategy="mean",
            include_final_norm=True,
        )
        embedder.eval()

        result = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)

        # Check result structure
        assert result is not None
        assert "embeddings" in result
        assert "seq_idx" in result

        # Check shapes
        assert result["embeddings"].shape == (2, 64)  # [B, H]
        assert result["seq_idx"].shape == (2,)

        # Check no pad_mask for mean pooling
        assert "pad_mask" not in result

        # Check embeddings are on CPU
        assert result["embeddings"].device == torch.device("cpu")

    def test_predict_step_per_token_pooling(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test predict_step with per_token returns pad_mask."""
        embedder = MockEmbedder(
            hidden_size=64,
            num_layers=2,
            pooling_strategy="per_token",
        )
        embedder.eval()

        result = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)

        # Check result structure
        assert "embeddings" in result
        assert "seq_idx" in result
        assert "pad_mask" in result  # Should be present for per_token

        # Check shapes
        assert result["embeddings"].shape == (2, 8, 64)  # [B, S, H]
        assert result["pad_mask"].shape == (2, 8)

    def test_predict_step_all_strategies(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test predict_step works with all pooling strategies."""
        strategies: list[PoolingStrategy] = ["mean", "max", "last", "first", "per_token"]

        for strategy in strategies:
            embedder = MockEmbedder(
                hidden_size=64,
                num_layers=2,
                pooling_strategy=strategy,
            )
            embedder.eval()

            result = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)

            assert result is not None, f"Strategy {strategy} returned None"
            assert "embeddings" in result, f"Strategy {strategy} missing embeddings"

            if strategy == "per_token":
                assert result["embeddings"].shape == (2, 8, 64)
                assert "pad_mask" in result
            else:
                assert result["embeddings"].shape == (2, 64)

    def test_embedding_layer_zero(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test that embedding_layer=0 returns embedding layer output without decoder."""
        embedder = MockEmbedder(
            hidden_size=64,
            num_layers=2,
            embedding_layer=0,
            pooling_strategy="mean",
        )
        embedder.eval()

        # Track if decoder was called
        decoder_called = False
        original_forward = embedder.module.decoder.forward

        def spy_decoder(*args, **kwargs):
            nonlocal decoder_called
            decoder_called = True
            return original_forward(*args, **kwargs)

        embedder.module.decoder.forward = spy_decoder

        result = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)

        # Verify decoder was not called
        assert not decoder_called, "Decoder should not be called when embedding_layer=0"

        # Verify result is valid
        assert result["embeddings"].shape == (2, 64)

    def test_empty_batch_returns_none(self, mock_parallel_state):
        """Test that empty batch returns None."""
        embedder = MockEmbedder(hidden_size=64, num_layers=2, pooling_strategy="mean")
        embedder.eval()

        result = embedder.predict_step({}, batch_idx=0, to_cpu=True)
        assert result is None

        result = embedder.predict_step(None, batch_idx=0, to_cpu=True)
        assert result is None

    def test_forward_for_embeddings_shape(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test that forward_for_embeddings returns correct shape."""
        embedder = MockEmbedder(hidden_size=64, num_layers=2, pooling_strategy="mean")
        embedder.eval()

        with torch.no_grad():
            hidden_states = embedder.forward_for_embeddings(sample_batch)

        # Should be [B, S, H]
        assert hidden_states.shape == (2, 8, 64)

    def test_determinism(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test that same input produces same output."""
        torch.manual_seed(42)
        embedder = MockEmbedder(hidden_size=64, num_layers=2, pooling_strategy="mean")
        embedder.eval()

        result1 = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)
        result2 = embedder.predict_step(sample_batch, batch_idx=0, to_cpu=True)

        torch.testing.assert_close(result1["embeddings"], result2["embeddings"])

    def test_include_final_norm_false(self, mock_parallel_state, sample_batch: Dict[str, Tensor]):
        """Test that include_final_norm=False skips final norm."""
        embedder_with_norm = MockEmbedder(
            hidden_size=64,
            num_layers=2,
            pooling_strategy="mean",
            include_final_norm=True,
        )
        embedder_with_norm.eval()

        embedder_without_norm = MockEmbedder(
            hidden_size=64,
            num_layers=2,
            pooling_strategy="mean",
            include_final_norm=False,
        )
        embedder_without_norm.eval()

        # Use same weights for fair comparison
        embedder_without_norm.module.load_state_dict(embedder_with_norm.module.state_dict())

        result_with = embedder_with_norm.predict_step(sample_batch, batch_idx=0, to_cpu=True)
        result_without = embedder_without_norm.predict_step(sample_batch, batch_idx=0, to_cpu=True)

        # Results should be different due to norm
        assert not torch.allclose(result_with["embeddings"], result_without["embeddings"])


# =============================================================================
# Additional Tests for Robustness
# =============================================================================


class TestRobustness:
    """Additional tests for robustness and edge cases."""

    def test_different_batch_sizes(self, sample_mask: Tensor):
        """Test pooling with various batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            hidden_states = torch.randn(batch_size, 4, 8)
            mask = torch.ones(batch_size, 4, dtype=torch.bool)

            result = _pool_hidden_states(hidden_states, mask, "mean")
            assert result.shape == (batch_size, 8)

    def test_different_sequence_lengths(self):
        """Test pooling with various sequence lengths."""
        for seq_len in [1, 4, 16, 64]:
            hidden_states = torch.randn(2, seq_len, 8)
            mask = torch.ones(2, seq_len, dtype=torch.bool)

            result = _pool_hidden_states(hidden_states, mask, "mean")
            assert result.shape == (2, 8)

    def test_different_hidden_sizes(self):
        """Test pooling with various hidden sizes."""
        for hidden_size in [8, 64, 256, 1024]:
            hidden_states = torch.randn(2, 4, hidden_size)
            mask = torch.ones(2, 4, dtype=torch.bool)

            result = _pool_hidden_states(hidden_states, mask, "mean")
            assert result.shape == (2, hidden_size)

    def test_float16_dtype(self):
        """Test pooling with float16 tensors."""
        hidden_states = torch.randn(2, 4, 8, dtype=torch.float16)
        mask = torch.ones(2, 4, dtype=torch.bool)

        result = _pool_hidden_states(hidden_states, mask, "mean")
        assert result.dtype == torch.float16
        assert result.shape == (2, 8)

    def test_bfloat16_dtype(self):
        """Test pooling with bfloat16 tensors."""
        hidden_states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        mask = torch.ones(2, 4, dtype=torch.bool)

        result = _pool_hidden_states(hidden_states, mask, "mean")
        assert result.dtype == torch.bfloat16
        assert result.shape == (2, 8)
