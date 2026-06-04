import mlx.core as mx
import numpy as np
import pytest

from mflux.utils.tensor_health import TensorHealth, TensorHealthError


def test_tensor_health_reports_mlx_context():
    tensor = mx.array([0.0, float("nan"), float("inf")], dtype=mx.float32)

    with pytest.raises(TensorHealthError) as exc:
        TensorHealth.ensure_finite(
            tensor,
            name="latents",
            phase="wan-scheduler-step",
            step=12,
            total_steps=40,
            timestep=812,
            denoiser="low",
            guidance=3.0,
        )

    report = exc.value.report
    assert report.non_finite_values == 2
    assert report.nan_values == 1
    assert report.inf_values == 1
    assert "step=12/40" in str(exc.value)
    assert "denoiser=low" in str(exc.value)


def test_tensor_health_mlx_failure_uses_scalar_diagnostics(monkeypatch):
    tensor = mx.array([0.0, float("nan"), 4.0], dtype=mx.float32)

    def fail_array(*args, **kwargs):
        raise AssertionError("MLX tensor health failure must not materialize the full tensor as NumPy.")

    monkeypatch.setattr(np, "array", fail_array)

    with pytest.raises(TensorHealthError) as exc:
        TensorHealth.ensure_finite(
            tensor,
            name="latents",
            phase="wan-scheduler-step",
        )

    report = exc.value.report
    assert report.non_finite_values == 1
    assert report.finite_min == 0.0
    assert report.finite_max == 4.0


def test_tensor_health_reports_numpy_frame_context():
    frame = np.zeros((2, 2, 3), dtype=np.float32)
    frame[0, 0, 0] = -np.inf

    with pytest.raises(TensorHealthError) as exc:
        TensorHealth.ensure_finite(
            frame,
            name="decoded_video_frame",
            phase="video_frame_conversion",
            frame=3,
            total_frames=8,
        )

    assert exc.value.report.inf_values == 1
    assert "frame=3/8" in str(exc.value)


def test_tensor_health_step_interval_checks_first_interval_and_final_step():
    assert TensorHealth.should_check_step(1, 40, 10)
    assert not TensorHealth.should_check_step(9, 40, 10)
    assert TensorHealth.should_check_step(10, 40, 10)
    assert TensorHealth.should_check_step(40, 40, 10)
    assert not TensorHealth.should_check_step(1, 40, 0)
