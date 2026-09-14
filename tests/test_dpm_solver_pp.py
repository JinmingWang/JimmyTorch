import inspect

import pytest
import torch

from DiffusionModels import DDIM, DPMSolverPP


def make_solver(order=2, inference_steps=4):
    return DPMSolverPP(
        min_beta=1e-3,
        max_beta=2e-2,
        max_diffusion_step=10,
        device="cpu",
        num_inference_steps=inference_steps,
        solver_order=order,
    )


def test_dpm_solver_configuration_and_generated_schedule():
    solver = make_solver(order=2, inference_steps=4)
    assert solver.samplingTimesteps() == (9, 6, 3, 0)
    assert "arxiv.org/abs/2211.01095" in inspect.getdoc(DPMSolverPP)

    with pytest.raises(ValueError, match="num_inference_steps"):
        DPMSolverPP(max_diffusion_step=10, num_inference_steps=11, device="cpu")
    with pytest.raises(ValueError, match="solver_order"):
        DPMSolverPP(solver_order=3, device="cpu")


@pytest.mark.parametrize(
    "timesteps, message",
    [
        ([8, 4, 0], "start"),
        ([9, 4, 1], "end"),
        ([9, 4, 4, 0], "strictly descending"),
        ([9, 10, 0], r"\[0, 9\]"),
    ],
)
def test_explicit_sampling_schedule_validation(timesteps, message):
    solver = make_solver()
    with pytest.raises(ValueError, match=message):
        solver.samplingTimesteps(timesteps)


def test_first_order_dpm_solver_step_matches_deterministic_ddim():
    solver = make_solver(order=1)
    ddim = DDIM(
        min_beta=solver.min_beta,
        max_beta=solver.max_beta,
        max_diffusion_step=solver.T,
        device="cpu",
        eta=0.0,
    )
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    current_timestep = 8
    previous_timestep = 3
    current_samples = solver.diffuse(clean_samples, current_timestep, epsilon_0_to_t)

    solver_result = solver.denoiseStep(
        current_samples,
        current_timestep,
        previous_timestep,
        epsilon_0_to_t_pred=epsilon_0_to_t,
    )
    ddim_result = ddim.denoiseStep(
        current_samples,
        current_timestep,
        previous_timestep,
        epsilon_0_to_t_pred=epsilon_0_to_t,
    )

    torch.testing.assert_close(solver_result, ddim_result, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "prediction_name", ("x0_pred", "epsilon_0_to_t_pred", "v_pred")
)
def test_dpm_solver_prediction_parameterizations_are_equivalent(prediction_name):
    solver = make_solver(order=1)
    clean_samples = torch.randn(2, 3, 4)
    epsilon_0_to_t = torch.randn_like(clean_samples)
    current_samples = solver.diffuse(clean_samples, 8, epsilon_0_to_t)
    predictions = {
        "x0_pred": clean_samples,
        "epsilon_0_to_t_pred": epsilon_0_to_t,
        "v_pred": solver.computeVelocity(clean_samples, epsilon_0_to_t, 8),
    }

    actual = solver.denoiseStep(
        current_samples,
        8,
        3,
        **{prediction_name: predictions[prediction_name]},
    )
    expected = solver.denoiseStep(current_samples, 8, 3, x0_pred=clean_samples)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_dpm_solver_clean_endpoint_returns_data_prediction():
    solver = make_solver()
    clean_prediction = torch.randn(2, 4)
    noisy_samples = torch.randn_like(clean_prediction)

    result, returned_prediction = solver.denoiseStep(
        noisy_samples,
        0,
        solver.clean_timestep,
        x0_pred=clean_prediction,
        need_x0=True,
    )

    torch.testing.assert_close(result, clean_prediction)
    torch.testing.assert_close(returned_prediction, clean_prediction)


def test_second_order_update_matches_multistep_formula():
    solver = make_solver(order=2)
    current_samples = torch.randn(2, 3, dtype=torch.float64)
    older_prediction = torch.randn_like(current_samples)
    current_prediction = torch.randn_like(current_samples)
    older_timestep, current_timestep, previous_timestep = 9, 6, 3

    actual = solver._secondOrderUpdate(
        current_samples,
        older_timestep,
        current_timestep,
        previous_timestep,
        older_prediction,
        current_prediction,
    )

    alpha_bars = solver.alpha_bars[:, 0].to(dtype=current_samples.dtype)
    alphas = torch.sqrt(alpha_bars)
    sigmas = torch.sqrt(1.0 - alpha_bars)
    lambdas = torch.log(alphas) - torch.log(sigmas)
    previous_lambda_step = lambdas[current_timestep] - lambdas[older_timestep]
    current_lambda_step = lambdas[previous_timestep] - lambdas[current_timestep]
    step_ratio = previous_lambda_step / current_lambda_step
    model_difference = (current_prediction - older_prediction) / step_ratio
    phi_1 = torch.expm1(-current_lambda_step)
    expected = (
        sigmas[previous_timestep] / sigmas[current_timestep] * current_samples
        - alphas[previous_timestep] * phi_1 * current_prediction
        - 0.5 * alphas[previous_timestep] * phi_1 * model_difference
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("solver_order", (1, 2))
def test_dpm_solver_callback_schedule_and_clean_reconstruction(solver_order):
    solver = make_solver(order=solver_order, inference_steps=4)
    clean_samples = torch.randn(2, 3, 4)
    initial_epsilon_0_to_t = torch.randn_like(clean_samples)
    initial_noise = solver.diffuse(clean_samples, 9, initial_epsilon_0_to_t)
    observed_timesteps = []

    def predict(noisy_samples, batch_timesteps):
        observed_timesteps.append(batch_timesteps.clone())
        return clean_samples, None, None

    result = solver.denoise(initial_noise, predict)

    torch.testing.assert_close(result, clean_samples)
    assert [values[0].item() for values in observed_timesteps] == [9, 6, 3, 0]
    assert all(values.shape == (2,) for values in observed_timesteps)


def test_dpm_solver_accepts_explicit_schedule_and_preserves_dtype():
    solver = make_solver(order=2)
    initial_noise = torch.randn(2, 4, dtype=torch.float64)

    result = solver.denoise(
        initial_noise,
        lambda samples, timesteps: (torch.zeros_like(samples), None, None),
        timesteps=[9, 7, 2, 0],
    )

    assert result.dtype == torch.float64
    assert result.shape == initial_noise.shape


def test_dpm_solver_and_ddim_can_use_equal_model_evaluation_counts():
    ddim = DDIM(max_diffusion_step=10, skip_step=3, device="cpu")
    ddim_schedule = list(range(ddim.T - 1, -1, -ddim.skip_step))
    if ddim_schedule[-1] != 0:
        ddim_schedule.append(0)
    solver = make_solver(order=2, inference_steps=len(ddim_schedule))
    initial_noise = torch.randn(2, 4)
    callback_counts = {"ddim": 0, "solver": 0}

    def ddim_predict(samples, batch_timesteps):
        callback_counts["ddim"] += 1
        return None, torch.zeros_like(samples), None

    def solver_predict(samples, batch_timesteps):
        callback_counts["solver"] += 1
        return None, torch.zeros_like(samples), None

    ddim.denoise(initial_noise, ddim_predict)
    solver.denoise(initial_noise, solver_predict)

    assert callback_counts == {"ddim": 4, "solver": 4}


def test_dpm_solver_prediction_contract():
    solver = make_solver()
    samples = torch.randn(2, 4)
    with pytest.raises(ValueError, match="exactly one"):
        solver.denoiseStep(samples, 1, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optional_cuda_dpm_solver_device_and_dtype_preservation():
    solver = DPMSolverPP(
        max_diffusion_step=4,
        num_inference_steps=2,
        device="cuda",
    )
    samples = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    result = solver.denoise(
        samples,
        lambda noisy_samples, timesteps: (
            torch.zeros_like(noisy_samples),
            None,
            None,
        ),
    )
    assert result.device.type == "cuda"
    assert result.dtype == samples.dtype
