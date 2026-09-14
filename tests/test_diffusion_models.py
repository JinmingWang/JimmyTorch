import inspect

import pytest
import torch

from DiffusionModels import DDIM, DDPM


PROCESS_TYPES = (DDPM, DDIM)


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
@pytest.mark.parametrize("scale_mode", ("linear", "quadratic", "log", "cosine"))
def test_schedules_are_valid_and_monotonic(process_type, scale_mode):
    process = process_type(
        min_beta=1e-4,
        max_beta=2e-2,
        max_diffusion_step=32,
        device="cpu",
        scale_mode=scale_mode,
    )

    assert process.betas.shape == (32, 1)
    assert process.alphas.shape == (32, 1)
    assert process.alpha_bars.shape == (32, 1)
    assert torch.all(process.betas > 0)
    assert torch.all(process.betas < 1)
    assert torch.all(process.alpha_bars[1:] < process.alpha_bars[:-1])


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_configuration_validation(process_type):
    with pytest.raises(ValueError, match="beta bounds"):
        process_type(min_beta=0, device="cpu")
    with pytest.raises(ValueError, match="max_diffusion_step"):
        process_type(max_diffusion_step=0, device="cpu")
    with pytest.raises(ValueError, match="scale_mode"):
        process_type(scale_mode="unknown", device="cpu")


def test_ddim_configuration_validation():
    with pytest.raises(ValueError, match="skip_step"):
        DDIM(skip_step=0, device="cpu")
    with pytest.raises(ValueError, match="eta"):
        DDIM(eta=-0.1, device="cpu")
    with pytest.raises(ValueError, match="eta"):
        DDIM(eta=1.1, device="cpu")


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_direct_and_single_step_diffusion_agree_at_first_transition(process_type):
    process = process_type(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(3, 2, 4, 5)
    epsilon_0_to_t = torch.randn_like(clean_samples)

    direct_samples = process.diffuse(clean_samples, 0, epsilon_0_to_t)
    stepped_samples = process.diffuseStep(clean_samples, 0, epsilon_0_to_t)

    torch.testing.assert_close(stepped_samples, direct_samples)


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_batched_timesteps_broadcast_without_flattening(process_type):
    process = process_type(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(3, 2, 4, 5, dtype=torch.float64)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    batch_timesteps = torch.tensor([0, 3, 7])

    noisy_samples = process.diffuse(clean_samples, batch_timesteps, epsilon_0_to_t)

    assert noisy_samples.shape == clean_samples.shape
    assert noisy_samples.dtype == clean_samples.dtype
    for batch_index, current_timestep in enumerate(batch_timesteps.tolist()):
        expected = process.diffuse(
            clean_samples[batch_index : batch_index + 1],
            current_timestep,
            epsilon_0_to_t[batch_index : batch_index + 1],
        )
        torch.testing.assert_close(
            noisy_samples[batch_index : batch_index + 1], expected
        )


def test_velocity_noise_and_velocity_extraction_round_trip():
    process = DDIM(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(3, 2, 5)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    batch_timesteps = torch.tensor([1, 4, 7])
    noisy_samples = process.diffuse(clean_samples, batch_timesteps, epsilon_0_to_t)

    extracted_noise = process.extractNoise(
        clean_samples, noisy_samples, batch_timesteps
    )
    expected_velocity = process.computeVelocity(
        clean_samples, epsilon_0_to_t, batch_timesteps
    )
    extracted_velocity = process.extractVelocity(
        clean_samples, noisy_samples, batch_timesteps
    )

    torch.testing.assert_close(extracted_noise, epsilon_0_to_t, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        extracted_velocity, expected_velocity, rtol=1e-4, atol=1e-5
    )


def test_ddpm_combined_noise_matches_two_forward_steps():
    process = DDPM(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(3, 2, 5)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    epsilon_t_to_tp1 = torch.randn_like(clean_samples)
    current_timestep = 4

    current_samples = process.diffuse(
        clean_samples, current_timestep - 1, epsilon_0_to_t
    )
    stepped_samples = process.diffuseStep(
        current_samples, current_timestep, epsilon_t_to_tp1
    )
    epsilon_0_to_tp1 = process.combineNoise(
        epsilon_0_to_t, epsilon_t_to_tp1, current_timestep
    )
    direct_samples = process.diffuse(clean_samples, current_timestep, epsilon_0_to_tp1)

    torch.testing.assert_close(stepped_samples, direct_samples, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        process.combineNoise(epsilon_0_to_t, epsilon_t_to_tp1, 0),
        epsilon_t_to_tp1,
    )


def test_ddpm_step_matches_posterior_formula_and_generator():
    process = DDPM(
        min_beta=1e-3,
        max_beta=1e-2,
        max_diffusion_step=8,
        device="cpu",
    )
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    current_timestep = 5
    current_samples = process.diffuse(clean_samples, current_timestep, epsilon_0_to_t)
    actual_generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)

    actual = process.denoiseStep(
        current_samples,
        current_timestep,
        current_timestep - 1,
        x0_pred=clean_samples,
        generator=actual_generator,
    )
    expected_noise = torch.randn(current_samples.shape, generator=expected_generator)
    expected_mean = (
        process.posterior_mean_x0_coefficients[current_timestep]
        * clean_samples.flatten(1)
        + process.posterior_mean_xt_coefficients[current_timestep]
        * current_samples.flatten(1)
    ).view_as(current_samples)
    expected = (
        expected_mean
        + torch.sqrt(process.posterior_variance[current_timestep]) * expected_noise
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "prediction_name", ("x0_pred", "epsilon_0_to_t_pred", "v_pred")
)
def test_ddpm_final_step_returns_predicted_clean_sample(prediction_name):
    process = DDPM(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    noisy_samples = process.diffuse(clean_samples, 0, epsilon_0_to_t)
    prediction = {
        "x0_pred": clean_samples,
        "epsilon_0_to_t_pred": epsilon_0_to_t,
        "v_pred": process.computeVelocity(clean_samples, epsilon_0_to_t, 0),
    }

    final_samples = process.denoiseStep(
        noisy_samples,
        0,
        process.clean_timestep,
        **{prediction_name: prediction[prediction_name]},
    )

    torch.testing.assert_close(final_samples, clean_samples, atol=2e-5, rtol=2e-5)


def test_ddpm_rejects_non_adjacent_reverse_transition():
    process = DDPM(max_diffusion_step=8, device="cpu")
    samples = torch.randn(2, 4)
    with pytest.raises(ValueError, match="adjacent"):
        process.denoiseStep(
            samples, 5, 2, epsilon_0_to_t_pred=torch.zeros_like(samples)
        )


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_direct_step_warns_for_ambiguous_zero_endpoint(process_type):
    process = process_type(max_diffusion_step=8, device="cpu")
    samples = torch.randn(2, 4)
    with pytest.warns(DeprecationWarning, match="first noisy schedule index"):
        process.denoiseStep(
            samples, 1, 0, epsilon_0_to_t_pred=torch.zeros_like(samples)
        )


def test_deterministic_ddim_matches_closed_form_skipped_step():
    process = DDIM(max_diffusion_step=8, device="cpu", eta=0)
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    current_timestep = 6
    previous_timestep = 2
    current_samples = process.diffuse(clean_samples, current_timestep, epsilon_0_to_t)

    actual = process.denoiseStep(
        current_samples,
        current_timestep,
        previous_timestep,
        epsilon_0_to_t_pred=epsilon_0_to_t,
    )
    expected = process.diffuse(clean_samples, previous_timestep, epsilon_0_to_t)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "prediction_name", ("x0_pred", "epsilon_0_to_t_pred", "v_pred")
)
def test_ddim_prediction_parameterizations_are_equivalent(prediction_name):
    process = DDIM(max_diffusion_step=8, device="cpu", eta=0)
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    current_timestep = 6
    previous_timestep = 2
    current_samples = process.diffuse(clean_samples, current_timestep, epsilon_0_to_t)
    predictions = {
        "x0_pred": clean_samples,
        "epsilon_0_to_t_pred": epsilon_0_to_t,
        "v_pred": process.computeVelocity(
            clean_samples, epsilon_0_to_t, current_timestep
        ),
    }

    actual, predicted_clean = process.denoiseStep(
        current_samples,
        current_timestep,
        previous_timestep,
        need_x0=True,
        **{prediction_name: predictions[prediction_name]},
    )
    expected = process.diffuse(clean_samples, previous_timestep, epsilon_0_to_t)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(predicted_clean, clean_samples, atol=2e-5, rtol=2e-5)


def test_stochastic_ddim_matches_eta_equation():
    process = DDIM(max_diffusion_step=8, device="cpu", eta=0.7)
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t_pred = torch.randn_like(clean_samples)
    current_timestep = 6
    previous_timestep = 2
    current_samples = process.diffuse(
        clean_samples, current_timestep, epsilon_0_to_t_pred
    )
    actual_generator = torch.Generator().manual_seed(29)
    expected_generator = torch.Generator().manual_seed(29)

    actual = process.denoiseStep(
        current_samples,
        current_timestep,
        previous_timestep,
        epsilon_0_to_t_pred=epsilon_0_to_t_pred,
        generator=actual_generator,
    )
    alpha_bar = process.alpha_bars[current_timestep]
    previous_alpha_bar = process.alpha_bars[previous_timestep]
    sigma = process.eta * torch.sqrt(
        (1 - previous_alpha_bar)
        / (1 - alpha_bar)
        * (1 - alpha_bar / previous_alpha_bar)
    )
    ddim_reverse_step_noise = torch.randn(
        current_samples.shape, generator=expected_generator
    )
    expected = (
        torch.sqrt(previous_alpha_bar) * clean_samples
        + torch.sqrt(1 - previous_alpha_bar - sigma.square()) * epsilon_0_to_t_pred
        + sigma * ddim_reverse_step_noise
    )

    torch.testing.assert_close(actual, expected)


def test_stochastic_ddim_is_reproducible_with_generator():
    process = DDIM(max_diffusion_step=8, device="cpu", eta=0.7)
    samples = torch.randn(2, 3, 4)
    epsilon_0_to_t_pred = torch.zeros_like(samples)
    first_generator = torch.Generator().manual_seed(23)
    second_generator = torch.Generator().manual_seed(23)

    first = process.denoiseStep(
        samples,
        6,
        2,
        epsilon_0_to_t_pred=epsilon_0_to_t_pred,
        generator=first_generator,
    )
    second = process.denoiseStep(
        samples,
        6,
        2,
        epsilon_0_to_t_pred=epsilon_0_to_t_pred,
        generator=second_generator,
    )

    torch.testing.assert_close(first, second)


def test_sampling_callbacks_receive_each_noise_level_once():
    ddpm = DDPM(max_diffusion_step=6, device="cpu")
    ddim = DDIM(max_diffusion_step=6, device="cpu", skip_step=2)
    initial_noise = torch.randn(2, 4)

    def collect(process):
        observed_timesteps = []

        def predict(samples, batch_timesteps):
            observed_timesteps.append(batch_timesteps[0].item())
            return None, torch.zeros_like(samples), None

        process.denoise(initial_noise, predict)
        return observed_timesteps

    assert collect(ddpm) == [5, 4, 3, 2, 1, 0]
    assert collect(ddim) == [5, 3, 1, 0]


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_prediction_contract_requires_exactly_one_prediction(process_type):
    process = process_type(max_diffusion_step=4, device="cpu")
    samples = torch.randn(2, 4)
    with pytest.raises(ValueError, match="exactly one"):
        process.denoiseStep(samples, 0, -1)
    with pytest.raises(ValueError, match="exactly one"):
        process.denoiseStep(
            samples,
            0,
            -1,
            x0_pred=samples,
            epsilon_0_to_t_pred=torch.zeros_like(samples),
        )


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_input_validation_rejects_bad_shape_dtype_and_timestep(process_type):
    process = process_type(max_diffusion_step=4, device="cpu")
    samples = torch.randn(2, 4)
    with pytest.raises(ValueError, match="shape"):
        process.diffuse(samples, 1, torch.randn(2, 5))
    with pytest.raises(TypeError, match="dtype"):
        process.diffuse(samples, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="timestep"):
        process.diffuse(samples, 4)


@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_internally_sampled_epsilon_0_to_t_is_reproducible(process_type):
    process = process_type(max_diffusion_step=4, device="cpu")
    samples = torch.zeros(2, 4)
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)
    first = process.diffuse(samples, 2, generator=first_generator)
    second = process.diffuse(samples, 2, generator=second_generator)
    torch.testing.assert_close(first, second)


def test_range_explicit_noise_keywords_and_legacy_aliases_agree():
    ddim = DDIM(max_diffusion_step=8, device="cpu")
    ddpm = DDPM(max_diffusion_step=8, device="cpu")
    clean_samples = torch.randn(2, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    epsilon_t_to_tp1 = torch.randn_like(clean_samples)

    canonical_diffusion = ddim.diffuse(clean_samples, 4, epsilon_0_to_t=epsilon_0_to_t)
    legacy_diffusion = ddim.diffuse(clean_samples, 4, noise=epsilon_0_to_t)
    torch.testing.assert_close(canonical_diffusion, legacy_diffusion)

    canonical_velocity = ddim.computeVelocity(
        clean_samples, epsilon_0_to_t=epsilon_0_to_t, t=4
    )
    legacy_velocity = ddim.computeVelocity(clean_samples, epsilon=epsilon_0_to_t, t=4)
    torch.testing.assert_close(canonical_velocity, legacy_velocity)

    canonical_step = ddim.denoiseStep(
        canonical_diffusion,
        4,
        2,
        epsilon_0_to_t_pred=epsilon_0_to_t,
    )
    legacy_step = ddim.denoiseStep(
        canonical_diffusion, 4, 2, epsilon_pred=epsilon_0_to_t
    )
    torch.testing.assert_close(canonical_step, legacy_step)

    canonical_combined = ddpm.combineNoise(
        epsilon_0_to_t=epsilon_0_to_t,
        epsilon_t_to_tp1=epsilon_t_to_tp1,
        t=4,
    )
    legacy_combined = ddpm.combineNoise(
        eps_0_to_t=epsilon_0_to_t,
        eps_t_to_tp1=epsilon_t_to_tp1,
        t=4,
    )
    torch.testing.assert_close(canonical_combined, legacy_combined)


def test_legacy_public_methods_and_attributes_remain_available():
    expected_ddpm_methods = {
        "diffuseStep",
        "diffuse",
        "computeVelocity",
        "denoiseStep",
        "denoise",
        "combineNoise",
    }
    expected_ddim_methods = {
        "diffuseStep",
        "diffuse",
        "denoiseStep",
        "denoise",
        "extractNoise",
        "computeVelocity",
        "extractVelocity",
    }
    for method_name in expected_ddpm_methods:
        assert callable(getattr(DDPM(device="cpu"), method_name))
    for method_name in expected_ddim_methods:
        assert callable(getattr(DDIM(device="cpu"), method_name))

    ddpm = DDPM(device="cpu")
    assert ddpm.beta is ddpm.betas
    assert ddpm.alpha is ddpm.alphas
    assert ddpm.αbar is ddpm.alpha_bars
    assert ddpm.sqrt_αbar is ddpm.sqrt_alpha_bars
    assert ddpm.sqrt_1_m_αbar is ddpm.sqrt_1m_alpha_bars
    assert "generator" in inspect.signature(ddpm.denoise).parameters
    assert "arxiv.org/abs/2006.11239" in inspect.getdoc(DDPM)
    assert "arxiv.org/abs/2010.02502" in inspect.getdoc(DDIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("process_type", PROCESS_TYPES)
def test_optional_cuda_dtype_and_device_preservation(process_type):
    process = process_type(max_diffusion_step=4, device="cuda")
    samples = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    result = process.diffuse(samples, torch.tensor([1, 2], device="cuda"))
    assert result.device.type == "cuda"
    assert result.dtype == samples.dtype
