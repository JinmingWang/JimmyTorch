from typing import Tuple

import numpy as np
import torch

Axis = Tuple[float, float, float]


def normalizeMesh(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center a mesh on its bounding box and scale its largest extent to one."""
    bounds_min = vertices.amin(dim=(0, 1))
    bounds_max = vertices.amax(dim=(0, 1))
    center = (bounds_min + bounds_max) / 2
    scale = (bounds_max - bounds_min).amax()
    return (vertices - center) / scale, center, scale


def denormalizePoints(points: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map normalized points back into their source STL coordinate frame."""
    return points * scale + center


def sampleAxis(generator: np.random.Generator) -> Axis:
    """Sample a unit vector uniformly from the surface of the sphere."""
    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)
    return tuple(float(value) for value in axis)


def buildPlaneBasis(axis: torch.Tensor) -> torch.Tensor:
    """Return a deterministic right-handed orthonormal basis spanning an axis-normal plane."""
    normal = axis / torch.linalg.vector_norm(axis)
    reference_index = torch.argmin(normal.abs())
    reference = torch.zeros(3, dtype=axis.dtype, device=axis.device)
    reference[reference_index] = 1
    first_axis = torch.linalg.cross(normal, reference, dim=0)
    first_axis = first_axis / torch.linalg.vector_norm(first_axis)
    second_axis = torch.linalg.cross(normal, first_axis, dim=0)
    return torch.stack((first_axis, second_axis))


def projectPointsTo2D(points: torch.Tensor, center: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project 3D points to coordinates relative to a plane center and basis."""
    return (points - center) @ basis.T


def reconstructPoints3D(points: torch.Tensor, center: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Reconstruct 3D points from plane-relative 2D coordinates."""
    return points @ basis + center