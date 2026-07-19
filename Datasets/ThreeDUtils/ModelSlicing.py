import torch
import numpy as np
from typing import List, Tuple, Optional
import warnings

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    warnings.warn("CUDA not available, using CPU. Performance will be poor for large meshes.")
    DEVICE = torch.device('cpu')

def meshSlicing(
    vertices: torch.Tensor,
    plane_interval: float,
    plane_axis: Optional[Tuple[float, float, float]] = None,
    initial_point: Optional[Tuple[float, float, float]] = None,
    eps: float = 1e-8
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    FULLY VECTORIZED slicing of a mesh with a *stack* of parallel planes.
    Every plane shares the same normal `plane_axis`, is spaced `plane_interval`
    apart from its neighbors, and the stack starts at `initial_point` and steps
    forward along `plane_axis` until it passes the far end of the mesh.
    No loops over triangles - all operations on GPU in parallel. The only
    Python-level loop is the final split of results into one entry per plane.

    Args:
        vertices: Tensor of shape (F, 3, 3) from loadSTL
        plane_interval: Positive gap between consecutive planes along plane_axis
        plane_axis: Shared normal direction of every slicing plane (nx, ny, nz). If None, defaults to (0, 0, 1), which means slicing from bottom to top along the Z-axis.
        initial_point: A point on the first plane (x, y, z). If None, the
            mesh's own extreme vertex along plane_axis is used - i.e. slicing
            starts from "the first point in the model along plane_axis".
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of:
        - indices_list: list of length P (one per plane); indices_list[p] is a
          (M_p,) tensor of triangle indices intersected by plane p
        - points_list: list of length P; points_list[p] is a (M_p, 2, 3) tensor
          of intersection points for plane p (each face contributes 2 points)
        - plane_points: Tensor of shape (P, 3), the 3D anchor point of each
          plane (useful for visualization)
    """
    if plane_interval <= 0:
        raise ValueError("plane_interval must be positive")
    
    if plane_axis is None:
        plane_axis = (0.0, 0.0, 1.0)  # Default to slicing from bottom to top along Z-axis

    # Match the dtype of the input vertices (e.g. float32 or float16) so callers
    # can trade precision for speed without touching this function.
    dtype = vertices.dtype

    axis = torch.tensor(plane_axis, dtype=dtype, device=DEVICE)
    axis = axis / (torch.norm(axis) + eps)

    # Extract vertices - shape (F, 3) each
    v0 = vertices[:, 0, :]  # (F, 3)
    v1 = vertices[:, 1, :]  # (F, 3)
    v2 = vertices[:, 2, :]  # (F, 3)
    F = v0.shape[0]

    # Project every vertex onto the shared axis once - every plane reuses this.
    proj0 = torch.sum(v0 * axis, dim=1)  # (F,)
    proj1 = torch.sum(v1 * axis, dim=1)  # (F,)
    proj2 = torch.sum(v2 * axis, dim=1)  # (F,)

    if initial_point is None:
        # "First point in the model along plane_axis": the mesh vertex with
        # the smallest projection, i.e. the extreme point in that direction.
        start_proj = torch.min(torch.stack([proj0.min(), proj1.min(), proj2.min()]))
    else:
        initial_point_t = torch.tensor(initial_point, dtype=dtype, device=DEVICE)
        start_proj = torch.sum(initial_point_t * axis)

    max_proj = torch.max(torch.stack([proj0.max(), proj1.max(), proj2.max()]))
    span = (max_proj - start_proj).item()
    n_planes = int(span // plane_interval) + 1 if span > 0 else 1

    # (P,) scalar offsets of each plane along the axis, and their 3D anchor points
    offsets = start_proj + torch.arange(n_planes, dtype=dtype, device=DEVICE) * plane_interval  # (P,)
    plane_points = offsets.unsqueeze(1) * axis.unsqueeze(0)  # (P, 3)

    # Since every plane shares the same normal, the signed distance of a vertex
    # to plane p is just its projection minus that plane's scalar offset -
    # fully vectorized across (plane, triangle) with no need to re-broadcast
    # raw vertex coordinates per plane.
    d0 = proj0.unsqueeze(0) - offsets.unsqueeze(1)  # (P, F)
    d1 = proj1.unsqueeze(0) - offsets.unsqueeze(1)  # (P, F)
    d2 = proj2.unsqueeze(0) - offsets.unsqueeze(1)  # (P, F)

    d_stack = torch.stack([d0, d1, d2], dim=-1)  # (P, F, 3)
    signs = torch.sign(d_stack)  # (P, F, 3)

    has_pos = (signs > 0).any(dim=-1)  # (P, F)
    has_neg = (signs < 0).any(dim=-1)  # (P, F)
    has_zero = (signs == 0).any(dim=-1)  # (P, F)

    intersect_mask = (has_pos & has_neg) | (has_zero & (has_pos | has_neg))  # (P, F)
    flat_idx = torch.where(intersect_mask.reshape(-1))[0]  # (T,) index into flattened (P, F)

    empty_indices = [torch.empty((0,), dtype=torch.long, device=DEVICE) for _ in range(n_planes)]
    empty_points = [torch.empty((0, 2, 3), dtype=dtype, device=DEVICE) for _ in range(n_planes)]

    T = len(flat_idx)
    if T == 0:
        return empty_indices, empty_points, plane_points

    plane_idx = torch.div(flat_idx, F, rounding_mode='floor')  # (T,)
    tri_idx = flat_idx % F  # (T,)

    # Filter to intersecting (plane, triangle) rows - VECTORIZED
    v0_i = v0[tri_idx]  # (T, 3)
    v1_i = v1[tri_idx]  # (T, 3)
    v2_i = v2[tri_idx]  # (T, 3)
    d0_i = d0.reshape(-1)[flat_idx]  # (T,)
    d1_i = d1.reshape(-1)[flat_idx]  # (T,)
    d2_i = d2.reshape(-1)[flat_idx]  # (T,)

    # ============================================================
    # VECTORIZED INTERSECTION COMPUTATION - NO LOOPS
    # (identical per-triangle math as the single-plane version, just applied
    # to the combined rows across all planes at once)
    # ============================================================

    # Determine which edges intersect the plane
    # Edge 0-1: intersects if d0 and d1 have different signs or one is zero
    edge01_mask = (d0_i * d1_i <= 0) & ((d0_i.abs() > eps) | (d1_i.abs() > eps))
    edge12_mask = (d1_i * d2_i <= 0) & ((d1_i.abs() > eps) | (d2_i.abs() > eps))
    edge20_mask = (d2_i * d0_i <= 0) & ((d2_i.abs() > eps) | (d0_i.abs() > eps))

    # Function to compute intersection on each edge - VECTORIZED
    def compute_intersection(p1, p2, d1, d2, mask):
        """
        Compute intersection points for edges where mask is True.
        Fully vectorized with broadcasting.
        """
        # Compute intersection parameter t = d1 / (d1 - d2)
        denom = d1 - d2
        # Use mask to avoid division by zero
        t = torch.zeros_like(d1)
        valid = mask & (denom.abs() > eps)
        t[valid] = d1[valid] / denom[valid]
        t = torch.clamp(t, 0.0, 1.0)

        # Compute intersection point: p = p1 + t * (p2 - p1)
        # Broadcasting: (T,) * (T, 3) -> (T, 3)
        intersection = p1 + t.unsqueeze(1) * (p2 - p1)

        # Where mask is False, set to zeros (will be filtered later)
        return intersection, mask

    # Compute intersections for all three edges - VECTORIZED
    p01, mask01 = compute_intersection(v0_i, v1_i, d0_i, d1_i, edge01_mask)
    p12, mask12 = compute_intersection(v1_i, v2_i, d1_i, d2_i, edge12_mask)
    p20, mask20 = compute_intersection(v2_i, v0_i, d2_i, d0_i, edge20_mask)

    # ============================================================
    # HANDLE CASE: Vertex lies exactly on the plane
    # ============================================================
    # For triangles with vertex on plane, we need to identify the vertex
    # and the intersection point on the opposite edge

    # Find which vertex is on the plane
    v0_on_plane = d0_i.abs() < eps  # (T,)
    v1_on_plane = d1_i.abs() < eps  # (T,)
    v2_on_plane = d2_i.abs() < eps  # (T,)

    # Count how many vertices are on the plane
    num_on_plane = v0_on_plane.int() + v1_on_plane.int() + v2_on_plane.int()

    # Cases:
    # - 0 vertices on plane: standard case, 2 edges intersect (already computed)
    # - 1 vertex on plane: vertex + opposite edge intersection
    # - 2+ vertices on plane: triangle lies on plane (degenerate)

    # Create output tensor for all (plane, triangle) rows
    points = torch.zeros((T, 2, 3), dtype=dtype, device=DEVICE)

    # CASE 1: Standard case (no vertex on plane)
    standard_case = (num_on_plane == 0)  # (T,)

    # For standard case, we need to collect the two intersection points
    # We can create a mask for each edge that contributes
    # Then collect the first two valid intersections
    all_points = torch.stack([p01, p12, p20], dim=1)  # (T, 3, 3)
    all_masks = torch.stack([mask01, mask12, mask20], dim=1)  # (T, 3)

    # We want to select the first two True in all_masks for each row -
    # use cumsum to find those positions
    cumsum_mask = all_masks.cumsum(dim=1)  # (T, 3)

    # Positions where cumsum is 1 (first intersection)
    first_pos = (cumsum_mask == 1) & all_masks  # (T, 3)
    # Positions where cumsum is 2 (second intersection)
    second_pos = (cumsum_mask == 2) & all_masks  # (T, 3)

    # Select points using these masks
    first_points = (all_points * first_pos.unsqueeze(2)).sum(dim=1)  # (T, 3)
    second_points = (all_points * second_pos.unsqueeze(2)).sum(dim=1)  # (T, 3)

    # Fill output for standard case
    points[standard_case, 0] = first_points[standard_case]
    points[standard_case, 1] = second_points[standard_case]

    # CASE 2: One vertex on plane
    one_vertex_case = (num_on_plane == 1) & ~standard_case

    # Find the vertex on the plane for each triangle
    vertex_on_plane = torch.zeros((T, 3), dtype=dtype, device=DEVICE)
    vertex_on_plane[v0_on_plane] = v0_i[v0_on_plane]
    vertex_on_plane[v1_on_plane] = v1_i[v1_on_plane]
    vertex_on_plane[v2_on_plane] = v2_i[v2_on_plane]

    # Get the intersection point on the opposite edge
    # If v0 on plane, opposite edge is 1-2; if v1, opposite is 0-2; if v2, opposite is 0-1
    opposite_points = torch.zeros_like(vertex_on_plane)
    opposite_points[v0_on_plane] = p12[v0_on_plane]
    opposite_points[v1_on_plane] = p20[v1_on_plane]
    opposite_points[v2_on_plane] = p01[v2_on_plane]

    # Fill output for one vertex on plane case
    points[one_vertex_case, 0] = vertex_on_plane[one_vertex_case]
    points[one_vertex_case, 1] = opposite_points[one_vertex_case]

    # CASE 3: Multiple vertices on plane (triangle lies in plane)
    # We'll just use the triangle's edges as the intersection
    multi_vertex_case = (num_on_plane >= 2) & ~standard_case & ~one_vertex_case

    # For these, the triangle itself is on the plane
    # Use any two vertices as the line segment
    points[multi_vertex_case, 0] = v0_i[multi_vertex_case]
    points[multi_vertex_case, 1] = v1_i[multi_vertex_case]

    # Filter out any cases where points are identical (degenerate)
    # Compute distance between the two points
    dist = (points[:, 0] - points[:, 1]).norm(dim=1)  # (T,)
    degenerate = dist < eps

    # Remove degenerate triangles (where intersection is a point, not a line)
    valid = ~degenerate
    final_tri_idx = tri_idx[valid]
    final_plane_idx = plane_idx[valid]
    final_points = points[valid]

    # Split the combined (plane, triangle) rows back into one list entry per
    # plane. This loop is over P (the number of planes, typically small), not
    # over triangles, so it stays cheap.
    indices_list = []
    points_list = []
    for p in range(n_planes):
        sel = final_plane_idx == p
        indices_list.append(final_tri_idx[sel])
        points_list.append(final_points[sel])

    return indices_list, points_list, plane_points
