import inspect

import pytest
import torch

from DiffusionModels import PredictorCorrectorSampler, VESDE, VPSDE


def test_score_sde_exported_classes_include_citations():
    for exported_class in (VPSDE, VESDE, PredictorCorrectorSampler):
        assert "arxiv.org/abs/2011.13456" in inspect.getdoc(exported_class)


def test_vpsde_analytic_drift_and_marginal_for_batched_times():
    sde = VPSDE(beta_min=0.1, beta_max=2.1, device="cpu")
    clean_samples = torch.ones(2, 3, 4, dtype=torch.float64)
    continuous_timesteps = torch.tensor([0.25, 0.75], dtype=torch.float64)

    drift, diffusion = sde.driftDiffusion(clean_samples, continuous_timesteps)
    beta_t = 0.1 + continuous_timesteps * 2.0
    expected_log_mean = (
        -0.25 * continuous_timesteps.square() * 2.0 - 0.5 * continuous_timesteps * 0.1
    )
    marginal_mean, marginal_std = sde.marginalProb(clean_samples, continuous_timesteps)

    torch.testing.assert_close(drift, -0.5 * beta_t[:, None, None] * clean_samples)
    torch.testing.assert_close(diffusion, torch.sqrt(beta_t))
    torch.testing.assert_close(
        marginal_mean,
        torch.exp(expected_log_mean)[:, None, None] * clean_samples,
    )
    torch.testing.assert_close(
        marginal_std, torch.sqrt(1.0 - torch.exp(2.0 * expected_log_mean))
    )


def test_vesde_analytic_drift_diffusion_and_marginal():
    sde = VESDE(sigma_min=0.01, sigma_max=10.0, device="cpu")
    clean_samples = torch.randn(3, 2, 5)
    continuous_timesteps = torch.tensor([0.0, 0.5, 1.0])
    expected_sigma = 0.01 * (10.0 / 0.01) ** continuous_timesteps

    drift, diffusion = sde.driftDiffusion(clean_samples, continuous_timesteps)
    marginal_mean, marginal_std = sde.marginalProb(clean_samples, continuous_timesteps)

    torch.testing.assert_close(drift, torch.zeros_like(clean_samples))
    torch.testing.assert_close(
        diffusion,
        expected_sigma * (2.0 * torch.log(torch.tensor(1000.0))).sqrt(),
    )
    torch.testing.assert_close(marginal_mean, clean_samples)
    torch.testing.assert_close(marginal_std, expected_sigma)


@pytest.mark.parametrize("sde", [VPSDE(device="cpu"), VESDE(device="cpu")])
def test_score_sde_diffusion_and_score_target_use_accumulated_noise(sde):
    clean_samples = torch.randn(2, 3, 4, dtype=torch.float64)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    continuous_timesteps = torch.tensor([0.2, 0.8], dtype=torch.float64)
    marginal_mean, marginal_std = sde.marginalProb(clean_samples, continuous_timesteps)

    noisy_samples = sde.diffuse(
        clean_samples,
        continuous_timesteps,
        epsilon_0_to_t=epsilon_0_to_t,
    )
    score_target = sde.computeScoreTarget(epsilon_0_to_t, continuous_timesteps)

    expected_noisy = marginal_mean + marginal_std[:, None, None] * epsilon_0_to_t
    expected_score = -epsilon_0_to_t / marginal_std[:, None, None]
    torch.testing.assert_close(noisy_samples, expected_noisy)
    torch.testing.assert_close(score_target, expected_score)
    assert noisy_samples.dtype == clean_samples.dtype


@pytest.mark.parametrize("sde", [VPSDE(device="cpu"), VESDE(device="cpu")])
def test_prior_sampling_is_reproducible(sde):
    first_generator = torch.Generator().manual_seed(13)
    second_generator = torch.Generator().manual_seed(13)
    first = sde.priorSampling((2, 3, 4), generator=first_generator, dtype=torch.float64)
    second = sde.priorSampling(
        (2, 3, 4), generator=second_generator, dtype=torch.float64
    )

    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float64


def test_score_sde_and_sampler_configuration_validation():
    with pytest.raises(ValueError, match="beta bounds"):
        VPSDE(beta_min=0, device="cpu")
    with pytest.raises(ValueError, match="sigma bounds"):
        VESDE(sigma_min=2, sigma_max=1, device="cpu")

    sde = VPSDE(device="cpu")
    with pytest.raises(ValueError, match="num_steps"):
        PredictorCorrectorSampler(sde, num_steps=0)
    with pytest.raises(ValueError, match="corrector_steps"):
        PredictorCorrectorSampler(sde, corrector_steps=-1)
    with pytest.raises(ValueError, match="snr"):
        PredictorCorrectorSampler(sde, snr=0)
    with pytest.raises(ValueError, match="epsilon"):
        PredictorCorrectorSampler(sde, epsilon=0)


def test_vp_score_target_rejects_zero_noise_endpoint():
    sde = VPSDE(device="cpu")
    with pytest.raises(ValueError, match="undefined"):
        sde.computeScoreTarget(torch.ones(2, 3), 0.0)


def test_predictor_only_sampler_uses_batch_times_and_one_score_per_step():
    sde = VPSDE(beta_min=0.1, beta_max=1.0, device="cpu")
    sampler = PredictorCorrectorSampler(
        sde, num_steps=4, corrector_steps=0, epsilon=0.1
    )
    initial_noise = torch.randn(3, 2, 4)
    observed_timesteps = []

    def score_func(noisy_samples, continuous_timesteps):
        observed_timesteps.append(continuous_timesteps.clone())
        return -noisy_samples

    result = sampler.denoise(
        initial_noise,
        score_func,
        generator=torch.Generator().manual_seed(17),
    )

    assert result.shape == initial_noise.shape
    assert torch.isfinite(result).all()
    assert len(observed_timesteps) == 4
    assert all(values.shape == (3,) for values in observed_timesteps)
    torch.testing.assert_close(
        torch.stack([values[0] for values in observed_timesteps]),
        torch.tensor([1.0, 0.775, 0.55, 0.325]),
    )


def test_predictor_corrector_sampler_is_seed_reproducible():
    sde = VESDE(sigma_min=0.1, sigma_max=2.0, device="cpu")
    sampler = PredictorCorrectorSampler(
        sde, num_steps=3, corrector_steps=1, snr=0.1, epsilon=0.05
    )
    initial_noise = torch.randn(2, 3, 4)

    def score_func(noisy_samples, continuous_timesteps):
        _, marginal_std = sde.marginalProb(
            torch.zeros_like(noisy_samples), continuous_timesteps
        )
        return -noisy_samples / marginal_std[:, None, None].square()

    first = sampler.denoise(
        initial_noise,
        score_func,
        generator=torch.Generator().manual_seed(19),
    )
    second = sampler.denoise(
        initial_noise,
        score_func,
        generator=torch.Generator().manual_seed(19),
    )

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_score_sde_step_validates_transition_and_score_shape():
    sde = VPSDE(device="cpu")
    sampler = PredictorCorrectorSampler(sde, corrector_steps=0)
    samples = torch.randn(2, 4)

    with pytest.raises(ValueError, match="strictly smaller"):
        sampler.denoiseStep(samples, 0.5, 0.5, lambda x, t: -x)
    with pytest.raises(ValueError, match="shape"):
        sampler.denoiseStep(samples, 0.5, 0.4, lambda x, t: torch.zeros(2, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optional_cuda_score_sde_device_and_dtype_preservation():
    sde = VPSDE(device="cuda")
    clean_samples = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    continuous_timesteps = torch.tensor([0.2, 0.8], device="cuda")
    result = sde.diffuse(clean_samples, continuous_timesteps)
    assert result.device.type == "cuda"
    assert result.dtype == clean_samples.dtype
