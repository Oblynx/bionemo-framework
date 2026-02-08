# Evo2 Embedding Extraction - Test Implementation Guide

**Purpose**: Guide for implementing comprehensive unit and integration tests
**Target**: CPU-based tests with tiny mock models for CI/CD
**Related Documents**:
- `evo2_embedding_impl_review.md` - Code review with issues found
- `evo2_embedding_design.md` - Original design specification
- `evo2_embedding_impl_reference.md` - Implementation reference

---

## Overview

This guide provides complete specifications for implementing tests for the Evo2 embedding extraction feature. All tests should run on **CPU with tiny mock models** to enable fast CI/CD execution.

**Key Constraint**: Tests must NOT require GPU or large checkpoints. Use small synthetic models (e.g., 2 layers, hidden_size=64).

---

## Implementation Context

### What Was Implemented

The following fixes have been implemented and need testing:

1. **`_unshuffle_zigzag()` function** (embedder.py:33-73)
   - Restores original sequence order from zigzag-packed tensors
   - Handles CP > 1 scenarios for position-sensitive pooling

2. **TP gathering** (embedder.py:145-152)
   - Gathers hidden states across Tensor Parallel ranks
   - Ensures complete embeddings when TP > 1

3. **Empty sequence validation** (embedder.py:286-295)
   - Validates sequences have valid tokens before pooling
   - Prevents `-inf` in max pooling and wrong positions in last pooling

4. **Five pooling strategies** (embedder.py:298-338)
   - `mean`: Average over valid tokens
   - `max`: Max over valid tokens
   - `last`: Last valid token
   - `first`: First token (position 0)
   - `per_token`: All tokens (masked)

5. **Integration with Context Parallelism** (embedder.py:162-166)
   - Automatic unshuffling for `last` and `per_token` with CP > 1

### What Was Fixed

Critical issues resolved that need verification:
- **No more `-inf` in max pooling** for empty sequences (now raises error)
- **No more wrong position in last pooling** for empty sequences (now raises error)
- **Zigzag ordering handled automatically** for CP > 1
- **TP gathering added** for models with sharded hidden dimensions

---

## Test File Structure

Create a new test file:
```
sub-packages/bionemo-evo2/tests/bionemo/evo2/test_embedder.py
```

### Required Imports

```python
import pytest
import torch
from torch import nn

from bionemo.evo2.models.embedder import (
    EmbeddingExtractorMixin,
    HyenaEmbedder,
    _pool_hidden_states,
    _unshuffle_zigzag,
)
```

---

## Test Suite 1: Unit Tests for `_unshuffle_zigzag()`

### Test 1.1: No-op for CP=1

**Purpose**: Verify that CP=1 returns input unchanged

```python
def test_unshuffle_zigzag_cp1_noop():
    """CP=1 should return input unchanged."""
    tensor = torch.randn(2, 8, 4)  # [B, S, H]
    result = _unshuffle_zigzag(tensor, cp_size=1, seq_dim=1)
    torch.testing.assert_close(result, tensor)
```

**Expected**: Tensor unchanged

---

### Test 1.2: CP=2 Unshuffling

**Purpose**: Verify correct unshuffling with CP=2

**Context**: With CP=2, sequence is split into 4 chunks in zigzag order:
- Rank 0 gets: [chunk_0, chunk_3]
- Rank 1 gets: [chunk_1, chunk_2]
- After gather: [chunk_0, chunk_3, chunk_1, chunk_2]
- Should restore to: [chunk_0, chunk_1, chunk_2, chunk_3]

```python
def test_unshuffle_zigzag_cp2():
    """Test unshuffling with CP=2 produces correct sequential order."""
    # Create tensor with zigzag order: [0, 0, 3, 3, 1, 1, 2, 2]
    # Each chunk is 2 elements
    zigzag_tensor = torch.tensor([[0, 0, 3, 3, 1, 1, 2, 2]], dtype=torch.float32)  # [1, 8]

    result = _unshuffle_zigzag(zigzag_tensor, cp_size=2, seq_dim=1)

    # Expected: [0, 0, 1, 1, 2, 2, 3, 3]
    expected = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)
```

**Expected**: Sequential order [0, 0, 1, 1, 2, 2, 3, 3]

---

### Test 1.3: CP=4 Unshuffling

**Purpose**: Verify correct unshuffling with larger CP

**Context**: With CP=4, 8 chunks in zigzag order:
- Ranks get: [0,7], [1,6], [2,5], [3,4]
- After gather: [0,7,1,6,2,5,3,4]
- Should restore to: [0,1,2,3,4,5,6,7]

```python
def test_unshuffle_zigzag_cp4():
    """Test unshuffling with CP=4."""
    # Zigzag order: [0,7,1,6,2,5,3,4]
    zigzag_tensor = torch.arange(8).float()[torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])].unsqueeze(0)

    result = _unshuffle_zigzag(zigzag_tensor, cp_size=4, seq_dim=1)

    # Expected: [0,1,2,3,4,5,6,7]
    expected = torch.arange(8).float().unsqueeze(0)
    torch.testing.assert_close(result, expected)
```

**Expected**: Sequential order [0,1,2,3,4,5,6,7]

---

### Test 1.4: Multiple Dimensions

**Purpose**: Verify unshuffling works on 3D tensors [B, S, H]

```python
def test_unshuffle_zigzag_3d_tensor():
    """Test unshuffling works correctly on [B, S, H] tensors."""
    batch_size = 2
    hidden_size = 4
    seq_len = 8

    # Create 3D tensor with zigzag pattern in seq dimension
    # For simplicity, use sequential values and manually zigzag
    sequential = torch.arange(batch_size * seq_len * hidden_size).float()
    sequential = sequential.reshape(batch_size, seq_len, hidden_size)

    # Manually create zigzag ordering for CP=2: [0,3,1,2]
    zigzag = torch.cat([
        sequential[:, 0:2, :],  # chunk 0
        sequential[:, 6:8, :],  # chunk 3
        sequential[:, 2:4, :],  # chunk 1
        sequential[:, 4:6, :],  # chunk 2
    ], dim=1)

    result = _unshuffle_zigzag(zigzag, cp_size=2, seq_dim=1)

    torch.testing.assert_close(result, sequential)
```

**Expected**: Restored to sequential order in sequence dimension

---

## Test Suite 2: Unit Tests for Pooling Strategies

### Fixtures

```python
@pytest.fixture
def sample_hidden_states():
    """Create sample hidden states [B=3, S=4, H=8]."""
    return torch.randn(3, 4, 8)


@pytest.fixture
def sample_mask():
    """Mask with varying sequence lengths.

    Batch 0: 3 valid tokens
    Batch 1: 2 valid tokens
    Batch 2: 4 valid tokens (all)
    """
    return torch.tensor([
        [True, True, True, False],   # length 3
        [True, True, False, False],  # length 2
        [True, True, True, True],    # length 4
    ])


@pytest.fixture
def empty_mask():
    """All-False mask for testing edge cases."""
    return torch.tensor([
        [False, False, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ])
```

---

### Test 2.1: Mean Pooling

**Purpose**: Verify mean pooling respects mask

```python
def test_mean_pooling(sample_hidden_states, sample_mask):
    """Mean pooling should compute mean only over valid tokens."""
    result = _pool_hidden_states(
        sample_hidden_states, sample_mask, "mean"
    )

    # Check shape
    assert result.shape == (3, 8), f"Expected shape (3, 8), got {result.shape}"

    # Verify computation for first batch item
    # Should be mean of positions 0, 1, 2 (not 3)
    expected_batch0 = sample_hidden_states[0, :3, :].mean(dim=0)
    torch.testing.assert_close(result[0], expected_batch0)

    # Verify for second batch item (positions 0, 1)
    expected_batch1 = sample_hidden_states[1, :2, :].mean(dim=0)
    torch.testing.assert_close(result[1], expected_batch1)
```

**Expected**: Mean computed only over valid (True) positions

---

### Test 2.2: Max Pooling

**Purpose**: Verify max pooling ignores padded positions

```python
def test_max_pooling(sample_hidden_states, sample_mask):
    """Max pooling should find max over valid tokens only."""
    result = _pool_hidden_states(
        sample_hidden_states, sample_mask, "max"
    )

    # Check shape
    assert result.shape == (3, 8)

    # Verify computation for first batch item
    expected_batch0 = sample_hidden_states[0, :3, :].max(dim=0).values
    torch.testing.assert_close(result[0], expected_batch0)

    # Check that padding positions don't affect result
    # (Set padding to very large values, should still get max from valid tokens)
    modified_hidden = sample_hidden_states.clone()
    modified_hidden[0, 3, :] = 1e6  # Huge value in padding position

    result_modified = _pool_hidden_states(modified_hidden, sample_mask, "max")
    torch.testing.assert_close(result_modified[0], expected_batch0)
```

**Expected**: Max from valid positions only, ignoring padding

---

### Test 2.3: Last Token Pooling

**Purpose**: Verify last pooling gets correct last valid token

```python
def test_last_pooling(sample_hidden_states, sample_mask):
    """Last pooling should get the last valid token."""
    result = _pool_hidden_states(
        sample_hidden_states, sample_mask, "last"
    )

    # Check shape
    assert result.shape == (3, 8)

    # Batch 0: last valid is position 2 (3rd token)
    torch.testing.assert_close(result[0], sample_hidden_states[0, 2, :])

    # Batch 1: last valid is position 1 (2nd token)
    torch.testing.assert_close(result[1], sample_hidden_states[1, 1, :])

    # Batch 2: last valid is position 3 (4th token, all valid)
    torch.testing.assert_close(result[2], sample_hidden_states[2, 3, :])
```

**Expected**: Last valid token per sequence

---

### Test 2.4: First Token Pooling

**Purpose**: Verify first pooling always gets position 0

```python
def test_first_pooling(sample_hidden_states, sample_mask):
    """First pooling should always get position 0."""
    result = _pool_hidden_states(
        sample_hidden_states, sample_mask, "first"
    )

    # Check shape
    assert result.shape == (3, 8)

    # Should be identical to [:, 0, :]
    expected = sample_hidden_states[:, 0, :]
    torch.testing.assert_close(result, expected)
```

**Expected**: Position 0 for all sequences

---

### Test 2.5: Per-Token Pooling

**Purpose**: Verify per-token returns masked sequence

```python
def test_per_token_pooling(sample_hidden_states, sample_mask):
    """Per-token returns full sequence with invalid positions zeroed."""
    result = _pool_hidden_states(
        sample_hidden_states, sample_mask, "per_token"
    )

    # Check shape (should preserve sequence dimension)
    assert result.shape == (3, 4, 8)

    # Check that valid positions are preserved
    # Batch 0, position 0 (valid)
    torch.testing.assert_close(result[0, 0, :], sample_hidden_states[0, 0, :])

    # Check that invalid positions are zeroed
    # Batch 0, position 3 (invalid)
    expected_zero = torch.zeros(8)
    torch.testing.assert_close(result[0, 3, :], expected_zero)

    # Batch 1, position 2 (invalid)
    torch.testing.assert_close(result[1, 2, :], expected_zero)
```

**Expected**: Full [B, S, H] tensor with masked positions zeroed

---

## Test Suite 3: Edge Cases and Error Handling

### Test 3.1: Empty Sequence with Mean Pooling

**Purpose**: Verify error is raised for empty sequences

```python
def test_empty_sequence_mean_raises_error(sample_hidden_states, empty_mask):
    """Mean pooling should raise ValueError for sequences with no valid tokens."""
    with pytest.raises(ValueError, match="no valid tokens"):
        _pool_hidden_states(sample_hidden_states, empty_mask, "mean")
```

**Expected**: `ValueError` with message about "no valid tokens"

---

### Test 3.2: Empty Sequence with Max Pooling

**Purpose**: Verify error is raised (no more `-inf` bug)

```python
def test_empty_sequence_max_raises_error(sample_hidden_states, empty_mask):
    """Max pooling should raise ValueError for empty sequences."""
    with pytest.raises(ValueError, match="no valid tokens"):
        _pool_hidden_states(sample_hidden_states, empty_mask, "max")
```

**Expected**: `ValueError` (not `-inf` values)

---

### Test 3.3: Empty Sequence with Last Pooling

**Purpose**: Verify error is raised (no more wrong position bug)

```python
def test_empty_sequence_last_raises_error(sample_hidden_states, empty_mask):
    """Last pooling should raise ValueError for empty sequences."""
    with pytest.raises(ValueError, match="no valid tokens"):
        _pool_hidden_states(sample_hidden_states, empty_mask, "last")
```

**Expected**: `ValueError` (not position 0)

---

### Test 3.4: Empty Sequence with Per-Token

**Purpose**: Verify per-token works even with empty sequences

```python
def test_empty_sequence_per_token_works(sample_hidden_states, empty_mask):
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
```

**Expected**: All-zero tensors for invalid positions (no error)

---

### Test 3.5: Invalid Pooling Strategy

**Purpose**: Verify error for unknown strategy

```python
def test_invalid_pooling_strategy(sample_hidden_states, sample_mask):
    """Should raise ValueError for unknown pooling strategy."""
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        _pool_hidden_states(sample_hidden_states, sample_mask, "invalid_strategy")
```

**Expected**: `ValueError` with "Unknown pooling strategy"

---

## Test Suite 4: Integration Tests (Mock Model Required)

### Mock Model Setup

Create a minimal mock model for testing the full `predict_step` flow:

```python
class MockMegatronModel(nn.Module):
    """Minimal mock of Megatron model for testing."""

    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pre_process = True
        self.post_process = True

        # Mock embedding
        self.embedding = nn.Embedding(512, hidden_size)

        # Mock decoder with layers
        self.decoder = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.decoder.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, position_ids=None):
        # Simple forward pass
        hidden = self.embedding(input_ids)
        hidden = hidden.transpose(0, 1)  # [S, B, H]

        for layer in self.decoder:
            hidden = layer(hidden)

        if self.decoder.final_norm:
            hidden = self.decoder.final_norm(hidden)

        return hidden


class MockEmbedder(EmbeddingExtractorMixin):
    """Mock embedder using mock Megatron model."""

    def __init__(self, hidden_size=64, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.module = MockMegatronModel(hidden_size, num_layers)
        self.tokenizer = None
        self.training = False
```

---

### Test 4.1: End-to-End with Mean Pooling

**Purpose**: Test full `predict_step` flow with mock model

```python
def test_predict_step_mean_pooling():
    """Test full predict_step with mean pooling."""
    # Create mock embedder
    embedder = MockEmbedder(
        hidden_size=64,
        num_layers=2,
        pooling_strategy="mean",
        include_final_norm=True
    )
    embedder.eval()

    # Create mock batch
    batch = {
        "tokens": torch.randint(0, 512, (2, 8)),  # [B=2, S=8]
        "position_ids": torch.arange(8).unsqueeze(0).expand(2, -1),
        "loss_mask": torch.tensor([
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, False, False],
        ]),
        "seq_idx": torch.tensor([0, 1]),
    }

    # Mock parallel state (CP=1, TP=1, PP=1)
    # You'll need to mock parallel_state methods to return appropriate values

    result = embedder.predict_step(batch, batch_idx=0, to_cpu=True)

    # Check result structure
    assert result is not None
    assert "embeddings" in result
    assert "seq_idx" in result

    # Check shapes
    assert result["embeddings"].shape == (2, 64)  # [B, H]
    assert result["seq_idx"].shape == (2,)

    # Check no pad_mask for mean pooling
    assert "pad_mask" not in result
```

**Note**: This test requires mocking `parallel_state` functions. See below for mock setup.

---

### Test 4.2: End-to-End with Per-Token Pooling

**Purpose**: Verify per-token includes pad_mask in result

```python
def test_predict_step_per_token_pooling():
    """Test predict_step with per_token returns pad_mask."""
    embedder = MockEmbedder(
        hidden_size=64,
        num_layers=2,
        pooling_strategy="per_token",
    )
    embedder.eval()

    batch = {
        "tokens": torch.randint(0, 512, (2, 8)),
        "position_ids": torch.arange(8).unsqueeze(0).expand(2, -1),
        "loss_mask": torch.ones(2, 8, dtype=torch.bool),
        "seq_idx": torch.tensor([0, 1]),
    }

    result = embedder.predict_step(batch, batch_idx=0, to_cpu=True)

    # Check result structure
    assert "embeddings" in result
    assert "seq_idx" in result
    assert "pad_mask" in result  # Should be present for per_token

    # Check shapes
    assert result["embeddings"].shape == (2, 8, 64)  # [B, S, H]
    assert result["pad_mask"].shape == (2, 8)
```

**Expected**: Result includes `pad_mask` for per-token strategy

---

### Test 4.3: Embedding Layer = 0

**Purpose**: Verify early exit when embedding_layer=0

```python
def test_embedding_layer_zero():
    """Test that embedding_layer=0 exits early without decoder."""
    embedder = MockEmbedder(
        hidden_size=64,
        num_layers=2,
        embedding_layer=0,
        pooling_strategy="mean",
    )
    embedder.eval()

    batch = {
        "tokens": torch.randint(0, 512, (2, 8)),
        "position_ids": torch.arange(8).unsqueeze(0).expand(2, -1),
        "loss_mask": torch.ones(2, 8, dtype=torch.bool),
        "seq_idx": torch.tensor([0, 1]),
    }

    # Spy on decoder to ensure it's not called
    decoder_called = False
    original_decoder_forward = embedder.module.decoder.forward
    def spy_decoder(*args, **kwargs):
        nonlocal decoder_called
        decoder_called = True
        return original_decoder_forward(*args, **kwargs)

    embedder.module.decoder.forward = spy_decoder

    result = embedder.predict_step(batch, batch_idx=0, to_cpu=True)

    # Verify decoder was not called
    assert not decoder_called, "Decoder should not be called when embedding_layer=0"

    # Verify result is valid
    assert result["embeddings"].shape == (2, 64)
```

**Expected**: Decoder not called, valid embeddings returned

---

## Mock Setup for Parallel State

For integration tests, you need to mock `parallel_state` functions:

```python
import unittest.mock as mock

@pytest.fixture
def mock_parallel_state_single_rank():
    """Mock parallel state for single-rank testing (CP=1, TP=1, PP=1)."""
    mocks = {
        'get_context_parallel_world_size': mock.patch(
            'megatron.core.parallel_state.get_context_parallel_world_size',
            return_value=1
        ),
        'get_tensor_model_parallel_world_size': mock.patch(
            'megatron.core.parallel_state.get_tensor_model_parallel_world_size',
            return_value=1
        ),
        'is_pipeline_last_stage': mock.patch(
            'megatron.core.parallel_state.is_pipeline_last_stage',
            return_value=True
        ),
    }

    for m in mocks.values():
        m.start()

    yield

    for m in mocks.values():
        m.stop()
```

Usage in tests:
```python
def test_with_mocks(mock_parallel_state_single_rank):
    # Test code here - parallel_state functions are mocked
    pass
```

---

## Test Organization

Organize tests with clear markers:

```python
# At the top of test_embedder.py
pytestmark = [
    pytest.mark.unit,  # Mark all as unit tests (no GPU needed)
]

# For specific categories
class TestPoolingStrategies:
    """Unit tests for pooling strategies."""
    pytestmark = pytest.mark.pooling

    # Tests go here...

class TestZigzagUnshuffling:
    """Unit tests for zigzag unshuffling."""
    pytestmark = pytest.mark.zigzag

    # Tests go here...

class TestIntegration:
    """Integration tests with mock model."""
    pytestmark = [pytest.mark.integration, pytest.mark.slow]

    # Tests go here...
```

---

## Running the Tests

```bash
# Run all embedder tests
pytest -v sub-packages/bionemo-evo2/tests/bionemo/evo2/test_embedder.py

# Run only pooling tests
pytest -v -m pooling sub-packages/bionemo-evo2/tests/bionemo/evo2/test_embedder.py

# Run fast tests only (skip slow integration tests)
pytest -v -m "not slow" sub-packages/bionemo-evo2/tests/bionemo/evo2/test_embedder.py

# Run with coverage
pytest --cov=bionemo.evo2.models.embedder sub-packages/bionemo-evo2/tests/bionemo/evo2/test_embedder.py
```

---

## Success Criteria

Tests are complete when:

- [ ] All 16 tests pass on CPU
- [ ] Code coverage > 90% for `embedder.py`
- [ ] All edge cases from review document are covered
- [ ] Tests run in < 10 seconds total
- [ ] No GPU or large checkpoint required
- [ ] pre-commit hooks pass

---

## Additional Test Ideas (Optional)

If time permits, add these tests:

1. **Batch size variations**: Test with batch_size=1, 2, 16
2. **Sequence length variations**: Test with seq_len=1, 8, 128
3. **Large hidden dimensions**: Test with hidden_size=1024
4. **All strategies on same input**: Verify different strategies produce expected differences
5. **Determinism**: Run same input twice, verify identical output
6. **Type handling**: Test with different dtypes (float32, float16, bfloat16)

---

## Notes for Implementation

### Import Considerations

```python
# Direct imports work in tests
from bionemo.evo2.models.embedder import (
    _pool_hidden_states,  # This is directly importable
    _unshuffle_zigzag,    # This is directly importable
)
```

### Handling Private Methods

The `_pool_hidden_states` and `_unshuffle_zigzag` functions are module-level functions (not methods), so they can be directly imported and tested.

### Mock Complexity

Keep mocks simple. For basic tests, you don't need full Megatron infrastructure - just mock the essential interfaces.

### CI/CD Considerations

- All tests must run on CPU
- Keep execution time < 10 seconds
- Use small models (hidden_size=64, num_layers=2)
- Mark slow tests with `@pytest.mark.slow` so they can be skipped if needed

---

## Reference: What Was Fixed

Remind yourself of the fixes that need verification:

| Fix | What Changed | What to Test |
|-----|--------------|--------------|
| Fix 1 | Added `_unshuffle_zigzag()` | CP=1,2,4 scenarios, 2D and 3D tensors |
| Fix 2 | Added TP gathering | TP > 1 produces complete embeddings |
| Fix 3 | Added empty sequence validation | Errors raised, not `-inf` or wrong positions |
| General | All pooling strategies | Correct computation, mask handling |

---

**End of Test Guide**

This document contains everything needed to implement comprehensive tests for the Evo2 embedding extraction feature.
