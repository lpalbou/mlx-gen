import mlx.core as mx
import numpy as np

from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.scheduler import WanEulerScheduler, WanUniPCMultistepScheduler


def test_wan_t2v_expanded_timesteps_match_diffusers_mask_policy():
    expanded = WanTimestepPolicy.expand_for_text_to_video(
        latent_shape=(2, 48, 3, 4, 6),
        timestep=937,
        patch_size=(1, 2, 2),
    )

    assert expanded.shape == (2, 18)
    assert np.all(np.array(expanded) == 937)


def test_wan_i2v_first_frame_timestep_mask_matches_diffusers_policy():
    mask = WanTimestepPolicy.first_frame_mask(latent_shape=(2, 48, 3, 4, 6))
    expanded = WanTimestepPolicy.expand_from_mask(mask=mask, batch_size=2, timestep=937, patch_size=(1, 2, 2))

    assert expanded.shape == (2, 18)
    expected = np.array([0] * 6 + [937] * 12, dtype=np.float32)
    np.testing.assert_array_equal(np.array(expanded[0]), expected)
    np.testing.assert_array_equal(np.array(expanded[1]), expected)


def test_wan_i2v_first_frame_condition_keeps_condition_frame_only():
    latents = mx.ones((1, 2, 3, 2, 2))
    condition = mx.zeros((1, 2, 3, 2, 2))
    first_frame_mask = WanTimestepPolicy.first_frame_mask(latent_shape=latents.shape)

    mixed = WanTimestepPolicy.apply_first_frame_condition(
        latents=latents,
        condition=condition,
        first_frame_mask=first_frame_mask,
    )

    mixed_np = np.array(mixed)
    assert np.all(mixed_np[:, :, 0] == 0)
    assert np.all(mixed_np[:, :, 1:] == 1)


def test_wan_unipc_flow_shift_5_timesteps_match_diffusers_reference():
    expected = {
        1: [999],
        2: [999, 833],
        3: [999, 909, 714],
        4: [999, 937, 833, 625],
        5: [999, 952, 882, 769, 556],
    }
    for steps, timesteps in expected.items():
        scheduler = WanUniPCMultistepScheduler()
        scheduler.set_timesteps(steps)
        assert np.array(scheduler.timesteps).tolist() == timesteps


def test_wan_unipc_flow_shift_5_sigmas_match_diffusers_reference():
    scheduler = WanUniPCMultistepScheduler()
    scheduler.set_timesteps(5)

    np.testing.assert_allclose(
        np.array(scheduler.sigmas),
        np.array(
            [
                0.9999989867210388,
                0.9524376392364502,
                0.8825258612632751,
                0.7696741223335266,
                0.55678790807724,
                0.0,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_wan_unipc_order2_flow_prediction_steps_match_diffusers_reference():
    scheduler = WanUniPCMultistepScheduler()
    scheduler.set_timesteps(4)
    sample = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10

    expected_sums = [
        27.45018768310547,
        26.876358032226562,
        25.032520294189453,
        19.023534774780273,
    ]
    expected_first_values = [
        [-0.006242090370506048, 0.09375791251659393, 0.19375790655612946],
        [-0.03015177696943283, 0.06984823197126389, 0.1698482185602188],
        [-0.10697832703590393, -0.006978313438594341, 0.09302167594432831],
        [-0.35735276341438293, -0.257352739572525, -0.1573527604341507],
    ]

    for index, timestep in enumerate(np.array(scheduler.timesteps).tolist()):
        model_output = mx.full(sample.shape, 0.1 * (index + 1), dtype=mx.float32)
        sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        np.testing.assert_allclose(float(mx.sum(sample).item()), expected_sums[index], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.array(sample.reshape(-1)[:3]),
            np.array(expected_first_values[index], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )


def test_wan_euler_flow_shift_5_timesteps_match_lightx2v_reference():
    scheduler = WanEulerScheduler()
    scheduler.set_timesteps(4)

    np.testing.assert_allclose(
        np.array(scheduler.timesteps),
        np.array([1000.0, 937.5, 833.3333, 625.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(scheduler.sigmas),
        np.array([1.0, 0.9375, 0.8333333, 0.625, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_wan_euler_steps_match_lightx2v_reference():
    scheduler = WanEulerScheduler()
    scheduler.set_timesteps(4)
    sample = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10

    expected_sums = [
        27.44999885559082,
        26.950000762939453,
        25.450000762939453,
        19.450000762939453,
    ]
    expected_first_values = [
        [-0.00625, 0.09375, 0.19375001],
        [-0.02708334, 0.07291666, 0.17291667],
        [-0.08958334, 0.01041667, 0.11041667],
        [-0.33958334, -0.23958333, -0.13958333],
    ]

    for index, timestep in enumerate(np.array(scheduler.timesteps).tolist()):
        model_output = mx.full(sample.shape, 0.1 * (index + 1), dtype=mx.float32)
        sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        np.testing.assert_allclose(float(mx.sum(sample).item()), expected_sums[index], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.array(sample.reshape(-1)[:3]),
            np.array(expected_first_values[index], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
