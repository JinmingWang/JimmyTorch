import torch
from rich.progress import track
from typing import Tuple

Tensor = torch.Tensor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pixelGrid(sdf_size: int, domain: Tuple[float, float], device, dtype) -> Tensor:
    """Return pixel-center coordinates as a (sdf_size * sdf_size, 2) tensor.

    Row-major layout with the outer axis running over y (row) and inner over x (column),
    so a reshape to (H, W) is a proper image with y increasing along the row axis.
    """
    lo, hi = domain
    step = (hi - lo) / sdf_size
    coords = torch.linspace(lo + 0.5 * step, hi - 0.5 * step, sdf_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)


def _unsignedDistance(
    grid_chunk: Tensor, p0: Tensor, p1: Tensor, d: Tensor, d_len_sq: Tensor, valid: Tensor,
) -> Tensor:
    """Analytic min point-to-segment distance for a (P_chunk, 2) pixel grid slice."""
    v = grid_chunk.unsqueeze(0).unsqueeze(0) - p0.unsqueeze(2)
    t = (v * d.unsqueeze(2)).sum(dim=-1) / d_len_sq.unsqueeze(-1)
    t = t.clamp(0.0, 1.0)
    closest = p0.unsqueeze(2) + t.unsqueeze(-1) * d.unsqueeze(2)
    dist = torch.linalg.vector_norm(grid_chunk.unsqueeze(0).unsqueeze(0) - closest, dim=-1)
    return dist.masked_fill(~valid.unsqueeze(-1), float("inf")).amin(dim=1)


def _rayCrossings(
    grid_chunk: Tensor, p0: Tensor, p1: Tensor, valid: Tensor, eps: float,
) -> Tensor:
    """Count horizontal-ray crossings from each pixel to the right through the segments."""
    gx = grid_chunk[:, 0]
    gy = grid_chunk[:, 1]
    y0 = p0[..., 1].unsqueeze(-1)
    y1 = p1[..., 1].unsqueeze(-1)
    x0 = p0[..., 0].unsqueeze(-1)
    x1 = p1[..., 0].unsqueeze(-1)
    straddles = (y0 <= gy) != (y1 <= gy)
    dy = y1 - y0
    dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)
    x_hit = x0 + (gy - y0) * (x1 - x0) / dy_safe
    crosses = straddles & (x_hit > gx) & valid.unsqueeze(-1)
    return crosses.sum(dim=1)


def computeSDFImages(
    segments_2d: Tensor,
    segment_counts: Tensor,
    sdf_size: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    chunk_size: int = 4,
    pixel_chunk_rows: int = 32,
    progress_description: str = "",
    device: torch.device = DEVICE,
    eps: float = 1e-12,
) -> Tensor:
    """Rasterize 2D SDF images from padded segment sets.

    Inside/outside sign is determined by a horizontal ray-crossing parity test on the
    segments, which correctly alternates through nested contours (outer negative,
    hole positive, deeper contour negative, etc.). The work is chunked over slices and
    over pixel-rows so peak VRAM stays roughly ``chunk_size * S_max * sdf_size *
    pixel_chunk_rows * 4B * (small constant)`` and never grows with the split's total
    slice count.

    Args:
        segments_2d: (N, S_max, 2, 2) padded segments in the SDF's coordinate frame.
        segment_counts: (N,) valid segment count per slice; padded slots ignored.
        sdf_size: Output grid resolution (H = W = sdf_size).
        domain: (min, max) extent covered by the pixel grid on both axes.
        chunk_size: Number of slices processed per GPU pass.
        pixel_chunk_rows: Number of pixel rows processed at a time within each slice chunk.
        progress_description: Rich progress label; empty disables progress output.
        device: CUDA device used for computation; result is always returned on CPU.
        eps: Guard against zero-length segments.

    Returns:
        Tensor of shape (N, sdf_size, sdf_size), float32 on CPU, negative inside contours.
    """
    if segments_2d.ndim != 4 or segments_2d.shape[-2:] != (2, 2):
        raise ValueError(f"segments_2d must have shape (N, S_max, 2, 2), got {segments_2d.shape}")
    n_slices, s_max = segments_2d.shape[:2]
    if segment_counts.shape != (n_slices,):
        raise ValueError(f"segment_counts must have shape ({n_slices},), got {segment_counts.shape}")

    dtype = torch.float32
    grid = _pixelGrid(sdf_size, domain, device, dtype)
    row_stride = sdf_size * max(1, min(pixel_chunk_rows, sdf_size))

    # Output kept on CPU: the full tensor for the train split can exceed GPU VRAM.
    output = torch.zeros((n_slices, sdf_size, sdf_size), dtype=dtype, device="cpu")
    if s_max == 0:
        return output

    seg_idx = torch.arange(s_max, device=device)
    chunk_starts = range(0, n_slices, chunk_size)
    if progress_description:
        chunk_starts = track(chunk_starts, total=(n_slices + chunk_size - 1) // chunk_size,
                             description=progress_description)
    for start in chunk_starts:
        end = min(start + chunk_size, n_slices)
        chunk = segments_2d[start:end].to(device=device, dtype=dtype)
        counts = segment_counts[start:end].to(device=device)
        if counts.max().item() == 0:
            continue

        c_size = end - start
        valid = seg_idx.unsqueeze(0) < counts.unsqueeze(1)

        p0 = chunk[..., 0, :]
        p1 = chunk[..., 1, :]
        d = p1 - p0
        d_len_sq = (d * d).sum(dim=-1).clamp_min(eps)

        unsigned = torch.empty((c_size, grid.shape[0]), dtype=dtype, device=device)
        crossings = torch.empty((c_size, grid.shape[0]), dtype=torch.int32, device=device)
        for p_start in range(0, grid.shape[0], row_stride):
            p_end = min(p_start + row_stride, grid.shape[0])
            grid_chunk = grid[p_start:p_end]
            unsigned[:, p_start:p_end] = _unsignedDistance(
                grid_chunk, p0, p1, d, d_len_sq, valid,
            )
            crossings[:, p_start:p_end] = _rayCrossings(
                grid_chunk, p0, p1, valid, eps,
            ).to(torch.int32)

        signed = torch.where((crossings % 2) == 1, -unsigned, unsigned)
        output[start:end] = signed.reshape(c_size, sdf_size, sdf_size).cpu()
        del chunk, counts, valid, p0, p1, d, d_len_sq, unsigned, crossings, signed

    return output
