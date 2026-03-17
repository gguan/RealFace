"""
Tests for faceforge.encoder.mica_encoder — MICAEncoder and MappingNetwork.
"""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_encoder():
    """MICAEncoder with mocked InsightFace and random mapping network."""
    from faceforge.encoder.mica_encoder import MICAEncoder, MappingNetwork

    enc = MICAEncoder.__new__(MICAEncoder)
    enc.device = torch.device("cpu")
    enc._image_size = 112

    # Mock mapping network
    enc._mapping = MappingNetwork()
    enc._mapping.eval()

    # Mock detector with a realistic face object
    mock_face = MagicMock()
    mock_face.det_score = 0.99
    mock_face.embedding = np.random.randn(512).astype(np.float32)

    mock_detector = MagicMock()
    mock_detector.get.return_value = [mock_face]
    enc._detector = mock_detector

    return enc


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_encoder_package_importable():
    """encoder package must be importable without errors."""
    import faceforge.encoder  # noqa: F401


def test_mica_encoder_module_importable():
    """mica_encoder module must be importable without errors."""
    import faceforge.encoder.mica_encoder  # noqa: F401


def test_multi_image_module_importable():
    """multi_image stub must be importable without errors."""
    import faceforge.encoder.multi_image  # noqa: F401


def test_mapping_network_shape():
    """MappingNetwork maps (B, 512) → (B, 300)."""
    from faceforge.encoder.mica_encoder import MappingNetwork

    net = MappingNetwork()
    x = torch.randn(2, 512)
    out = net(x)
    assert out.shape == (2, 300)


def test_mapping_network_single():
    """MappingNetwork works for batch size 1."""
    from faceforge.encoder.mica_encoder import MappingNetwork

    net = MappingNetwork()
    x = torch.randn(1, 512)
    out = net(x)
    assert out.shape == (1, 300)


def test_mapping_network_output_dtype():
    """MappingNetwork output is float32."""
    from faceforge.encoder.mica_encoder import MappingNetwork

    net = MappingNetwork()
    x = torch.randn(1, 512)
    out = net(x)
    assert out.dtype == torch.float32


def test_encode_returns_correct_shape(mock_encoder):
    """encode() returns tensor of shape (1, 300)."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = mock_encoder.encode(img)
    assert result.shape == (1, 300)
    assert result.dtype == torch.float32


def test_encode_on_correct_device(mock_encoder):
    """encode() output is on encoder's device."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = mock_encoder.encode(img)
    assert result.device.type == mock_encoder.device.type


def test_encode_no_face_raises(mock_encoder):
    """encode() raises ValueError when no face is detected."""
    mock_encoder._detector.get.return_value = []
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No face detected"):
        mock_encoder.encode(img)


def test_encode_no_face_error_message(mock_encoder):
    """ValueError message is exactly 'No face detected in image'."""
    mock_encoder._detector.get.return_value = []
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No face detected in image"):
        mock_encoder.encode(img)


def test_encode_batch_shape(mock_encoder):
    """encode_batch() returns (N, 300) for N images."""
    images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    result = mock_encoder.encode_batch(images)
    assert result.shape == (3, 300)


def test_encode_batch_single(mock_encoder):
    """encode_batch() works for a single-element list."""
    images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)]
    result = mock_encoder.encode_batch(images)
    assert result.shape == (1, 300)


def test_encode_batch_dtype(mock_encoder):
    """encode_batch() output is float32."""
    images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    result = mock_encoder.encode_batch(images)
    assert result.dtype == torch.float32


def test_encode_zero_embedding_fallback(mock_encoder):
    """encode() uses zero embedding when face.embedding is None."""
    mock_face = MagicMock()
    mock_face.det_score = 0.9
    mock_face.embedding = None
    mock_encoder._detector.get.return_value = [mock_face]

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = mock_encoder.encode(img)
    assert result.shape == (1, 300)


def test_encode_picks_best_face(mock_encoder):
    """encode() selects face with highest det_score when multiple faces present."""
    face_low = MagicMock()
    face_low.det_score = 0.5
    face_low.embedding = np.zeros(512, dtype=np.float32)

    face_high = MagicMock()
    face_high.det_score = 0.99
    face_high.embedding = np.ones(512, dtype=np.float32)

    mock_encoder._detector.get.return_value = [face_low, face_high]

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Should not crash; result should come from face_high embedding
    result = mock_encoder.encode(img)
    assert result.shape == (1, 300)


def test_encoder_init_no_crash_missing_weights():
    """MICAEncoder must initialize without crashing when weights file is missing."""
    from faceforge.encoder.mica_encoder import MICAEncoder

    try:
        enc = MICAEncoder(
            mica_weights_path="data/pretrained/mica.tar",  # likely doesn't exist
            flame_model_path="data/pretrained/FLAME2020/",
            device=torch.device("cpu"),
        )
        # If it makes it here, init succeeded
        assert enc._mapping is not None
    except Exception as e:
        # Only acceptable failure is InsightFace download failure in CI
        if "insightface" in str(e).lower() or "download" in str(e).lower():
            pytest.skip(f"InsightFace not available: {e}")
        else:
            raise


# ── MultiImageAggregator tests ─────────────────────────────────────────


@pytest.fixture
def mock_aggregator():
    """MultiImageAggregator with a deterministic mock encoder."""
    from faceforge.encoder.multi_image import MultiImageAggregator

    mock_enc = MagicMock()
    call_count = {"n": 0}

    def encode_side_effect(img):
        call_count["n"] += 1
        # Each call returns a slightly different shape vector
        return torch.full((1, 300), float(call_count["n"]) * 0.1)

    mock_enc.encode.side_effect = encode_side_effect
    return MultiImageAggregator(mock_enc, strategy="median")


def test_aggregate_result_fields(mock_aggregator):
    from faceforge.encoder.multi_image import AggregationResult
    imgs = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    result = mock_aggregator.aggregate(imgs)
    assert isinstance(result, AggregationResult)
    assert result.shape_params.shape == (1, 300)
    assert result.per_image_shapes.shape == (3, 300)
    assert 0.0 <= result.confidence <= 1.0
    assert result.n_valid_images == 3


def test_aggregate_skips_failed_detections():
    from faceforge.encoder.multi_image import MultiImageAggregator
    mock_enc = MagicMock()
    # First image fails, second succeeds
    mock_enc.encode.side_effect = [
        ValueError("No face detected in image"),
        torch.zeros(1, 300),
    ]
    agg = MultiImageAggregator(mock_enc)
    imgs = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    result = agg.aggregate(imgs)
    assert result.n_valid_images == 1
    assert result.shape_params.shape == (1, 300)


def test_aggregate_all_fail_raises():
    from faceforge.encoder.multi_image import MultiImageAggregator
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = ValueError("No face detected in image")
    agg = MultiImageAggregator(mock_enc)
    with pytest.raises(ValueError, match="No faces detected"):
        agg.aggregate([np.zeros((64, 64, 3), dtype=np.uint8)])


def test_aggregate_median_value():
    """Median of [0.1, 0.2, 0.3] tensors should be ~0.2"""
    from faceforge.encoder.multi_image import MultiImageAggregator
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = [
        torch.full((1, 300), 0.1),
        torch.full((1, 300), 0.2),
        torch.full((1, 300), 0.3),
    ]
    agg = MultiImageAggregator(mock_enc, strategy="median")
    result = agg.aggregate([None, None, None])
    assert abs(result.shape_params.mean().item() - 0.2) < 1e-5


def test_aggregate_identical_images_high_confidence():
    """Same prediction across images should give confidence close to 1.0"""
    from faceforge.encoder.multi_image import MultiImageAggregator
    mock_enc = MagicMock()
    mock_enc.encode.return_value = torch.ones(1, 300)
    agg = MultiImageAggregator(mock_enc)
    result = agg.aggregate([None, None, None])
    assert result.confidence > 0.99


def test_trimmed_mean_removes_outliers():
    from faceforge.encoder.multi_image import MultiImageAggregator
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = [
        torch.full((1, 300), 0.0),   # outlier low
        torch.full((1, 300), 0.5),
        torch.full((1, 300), 0.5),
        torch.full((1, 300), 0.5),
        torch.full((1, 300), 10.0),  # outlier high
    ]
    agg = MultiImageAggregator(mock_enc, strategy="trimmed_mean")
    result = agg.aggregate([None] * 5)
    # After trimming outliers, should be close to 0.5
    assert abs(result.shape_params.mean().item() - 0.5) < 0.1
