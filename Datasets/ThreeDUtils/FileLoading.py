import torch
import warnings
from stl import mesh
import numpy as np
from typing import List

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    warnings.warn("CUDA not available, using CPU. Performance will be poor for large meshes.")
    DEVICE = torch.device('cpu')

def loadSTL(path: str) -> torch.Tensor:
    """Load STL file and return vertices as a torch.float32 tensor on DEVICE."""
    stl_mesh = mesh.Mesh.from_file(path)
    vertices = stl_mesh.vectors  # Shape: (F, 3, 3), dtype already <f4 (float32)
    return torch.from_numpy(vertices.astype(np.float32)).to(DEVICE)


def _faces_to_tensor(vertices: np.ndarray, faces: List[List[int]]) -> torch.Tensor:
    """
    Fan-triangulate arbitrary polygon faces (quads, ngons, ...) and gather
    them into the same (F, 3, 3) per-triangle vertex layout that loadSTL
    returns, as a torch.float32 tensor on DEVICE.
    """
    triangles = [
        (face[0], face[i], face[i + 1])
        for face in faces
        for i in range(1, len(face) - 1)
    ]
    triangle_vertices = vertices[np.array(triangles, dtype=np.int64).reshape(-1, 3)]  # (F, 3, 3)
    return torch.from_numpy(triangle_vertices.astype(np.float32)).to(DEVICE)


def loadOFF(path: str) -> torch.Tensor:
    """Load an OFF (Object File Format) mesh and return vertices as a torch.float32 tensor on DEVICE."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    header = lines[0]
    if header == 'OFF':
        counts_line = lines[1]
        data_start = 2
    else:
        # Some OFF files (e.g. ModelNet) omit the newline after "OFF",
        # e.g. "OFF1413 1019 0".
        counts_line = header[3:]
        data_start = 1

    n_vertices, n_faces, _ = (int(x) for x in counts_line.split()[:3])

    vertex_lines = lines[data_start:data_start + n_vertices]
    vertices = np.array(
        [[float(x) for x in line.split()[:3]] for line in vertex_lines],
        dtype=np.float32,
    )

    face_lines = lines[data_start + n_vertices:data_start + n_vertices + n_faces]
    faces = []
    for line in face_lines:
        tokens = line.split()
        count = int(tokens[0])
        # Some OFF faces are followed by per-face color values after the
        # vertex indices - only take the first `count` indices.
        faces.append([int(tok) for tok in tokens[1:1 + count]])

    return _faces_to_tensor(vertices, faces)


def loadOBJ(path: str) -> torch.Tensor:
    """Load a Wavefront OBJ mesh and return vertices as a torch.float32 tensor on DEVICE."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            tag = tokens[0]
            if tag == 'v':
                vertices.append([float(x) for x in tokens[1:4]])
            elif tag == 'f':
                # Each token can be "v", "v/vt", "v/vt/vn" or "v//vn" - only
                # the vertex index (first field) matters for geometry.
                n = len(vertices)
                indices = []
                for tok in tokens[1:]:
                    idx = int(tok.split('/')[0])
                    # OBJ indices are 1-based; negative indices count backward
                    # from the current (already-parsed) vertex count.
                    indices.append(idx - 1 if idx > 0 else n + idx)
                faces.append(indices)

    vertices_np = np.array(vertices, dtype=np.float32)
    return _faces_to_tensor(vertices_np, faces)


def loadPLY(path: str) -> torch.Tensor:
    """Load an ASCII PLY mesh and return vertices as a torch.float32 tensor on DEVICE."""
    with open(path, 'r', errors='ignore') as f:
        raw_lines = f.readlines()

    if not raw_lines or raw_lines[0].strip() != 'ply':
        raise ValueError(f"Not a valid PLY file: {path}")

    format_line = next(line for line in raw_lines if line.startswith('format'))
    if 'ascii' not in format_line:
        raise NotImplementedError(
            "Only ASCII PLY files are supported. Binary PLY needs a dedicated "
            "struct-based parser (e.g. via the 'plyfile' or 'trimesh' package)."
        )

    n_vertices = 0
    n_faces = 0
    vertex_props: List[str] = []
    element = None
    header_end = 0
    for i, line in enumerate(raw_lines):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'element':
            element = tokens[1]
            if element == 'vertex':
                n_vertices = int(tokens[2])
            elif element == 'face':
                n_faces = int(tokens[2])
        elif tokens[0] == 'property' and element == 'vertex':
            vertex_props.append(tokens[-1])
        elif tokens[0] == 'end_header':
            header_end = i + 1
            break

    x_idx, y_idx, z_idx = vertex_props.index('x'), vertex_props.index('y'), vertex_props.index('z')

    vertex_lines = raw_lines[header_end:header_end + n_vertices]
    vertices = np.array(
        [[float(tok) for tok in line.split()] for line in vertex_lines],
        dtype=np.float32,
    )[:, [x_idx, y_idx, z_idx]]

    face_lines = raw_lines[header_end + n_vertices:header_end + n_vertices + n_faces]
    faces = []
    for line in face_lines:
        tokens = line.split()
        count = int(tokens[0])
        faces.append([int(tok) for tok in tokens[1:1 + count]])

    return _faces_to_tensor(vertices, faces)