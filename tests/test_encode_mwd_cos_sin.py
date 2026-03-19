from __future__ import annotations

import numpy as np

from scripts.encode_mwd_cos_sin import encode_angle_to_cos_sin, main, to_hw_float32


def test_to_hw_float32_replaces_none_and_squeezes_singleton_channel():
    sample = np.array([[[0.0], [90.0]], [[None], [360.0]]], dtype=object)

    output = to_hw_float32(sample)

    assert output.shape == (2, 2)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, np.array([[0.0, 90.0], [0.0, 360.0]], dtype=np.float32))


def test_encode_angle_to_cos_sin_uses_degree_semantics():
    angle_hw = np.array([[0.0, 90.0], [180.0, 360.0]], dtype=np.float32)

    encoded = encode_angle_to_cos_sin(angle_hw, angle_unit="degrees")

    assert encoded.shape == (2, 2, 2)
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded[0], np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(encoded[1], np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_main_encodes_directory(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    np.save(input_dir / "sample.npy", np.array([[0.0, 180.0], [90.0, 270.0]], dtype=np.float32))

    monkeypatch.setattr(
        "sys.argv",
        [
            "encode_mwd_cos_sin.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    encoded = np.load(output_dir / "sample.npy")
    assert encoded.shape == (2, 2, 2)
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded[0], np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(encoded[1], np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32), atol=1e-6)
