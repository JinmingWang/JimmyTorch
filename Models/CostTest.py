import torch
import torch.nn as nn
from calflops import calculate_flops


def testCudaSpeed(model: nn.Module, input_shapes: list[list[int]], output_shapes:list[list[int]]) -> tuple[float, float]:
    """
    Measure the speed of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be tested.
        input_shapes (list[list[int]]): A list of shapes for the input tensors.
        output_shapes (list[list[int]]): A list of shapes for the output tensors.
    """

    print(f"- Running speed test with {model.__class__.__name__} on CUDA...")

    model.cuda()

    input_tensors = [
        torch.randn(shape).cuda() for shape in input_shapes
    ]

    ground_truth_tensors = [
        torch.randn(shape).cuda() for shape in output_shapes
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training Speed Test

    # Warm up
    with torch.no_grad():
        for _ in range(200):
            _ = model(*input_tensors)

    # Measure time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)


    # Training
    start_time.record()
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(*input_tensors)
        loss = nn.MSELoss()(output, *ground_truth_tensors)
        loss.backward()
        optimizer.step()
    end_time.record()
    torch.cuda.synchronize()
    train_time = start_time.elapsed_time(end_time)

    # Evaluation Speed Test
    start_time.record()
    with torch.no_grad():
        for _ in range(1000):
            output = model(*input_tensors)
    end_time.record()
    torch.cuda.synchronize()
    eval_time = start_time.elapsed_time(end_time)

    # Calculate average time per iteration
    avg_train_time = train_time / 1000
    avg_eval_time = eval_time / 1000
    # print ms/iter and s/1000iter
    print(f"Training Time: {avg_train_time:.4f} ms/iter, or {avg_train_time:.4f} s/1000iter")
    print(f"Evaluation Time: {avg_eval_time:.4f} ms/iter, or {avg_eval_time:.4f} s/1000iter")

    return avg_train_time, avg_eval_time


def testFlops(model: nn.Module, input_shapes: list[list[int]], print_detailed: bool=False) -> None:
    """
    Measure the FLOPs of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be tested.
        input_shapes (list[list[int]]): A list of shapes for the input tensors.
    """

    print(f"Running FLOPs test with {model.__class__.__name__}...")

    input_tensors = [
        torch.randn(shape).cuda() for shape in input_shapes
    ]

    # Calculate FLOPs
    calculate_flops(model, args=input_tensors, print_detailed=print_detailed)