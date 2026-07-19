from pathlib import Path
from typing import List, Optional, Tuple
import argparse

import numpy as np
import torch
import plotly.graph_objects as go

from .FileLoading import loadSTL, loadOFF, loadOBJ, loadPLY
from .ModelSlicing import meshSlicing

_LOADERS = {
	".stl": loadSTL,
	".off": loadOFF,
	".obj": loadOBJ,
	".ply": loadPLY,
}


def loadModel(path: str) -> torch.Tensor:
	"""Load a 3D model file into a (F, 3, 3) tensor, dispatching on file extension."""
	suffix = Path(path).suffix.lower()
	loader = _LOADERS.get(suffix)
	if loader is None:
		raise ValueError(f"Unsupported model extension '{suffix}' for {path}")
	return loader(path)


def build_mesh_trace(vertices: torch.Tensor) -> go.Mesh3d:
	"""Build a Plotly Mesh3d trace from a (F, 3, 3) triangle-vertex tensor."""
	triangles = vertices.detach().cpu().numpy()

	# Flatten triangle vertices into point arrays expected by Plotly Mesh3d.
	x = triangles[:, :, 0].reshape(-1)
	y = triangles[:, :, 1].reshape(-1)
	z = triangles[:, :, 2].reshape(-1)

	# Triangle indices: every 3 vertices form one face.
	i = np.arange(0, len(x), 3)
	j = i + 1
	k = i + 2

	# Per-face intensity helps make neighboring faces visually distinguishable.
	face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
	norm_lengths = np.linalg.norm(face_normals, axis=1)
	norm_lengths[norm_lengths == 0] = 1.0
	normalized = face_normals / norm_lengths[:, None]
	face_intensity = np.abs(normalized).sum(axis=1)

	return go.Mesh3d(
		x=x,
		y=y,
		z=z,
		i=i,
		j=j,
		k=k,
		intensity=face_intensity,
		intensitymode="cell",
		colorscale="Turbo",
		opacity=1.0,
		flatshading=True,
		showscale=False,
		lighting=dict(ambient=0.38, diffuse=0.75, roughness=0.65, specular=0.25),
		lightposition=dict(x=100, y=200, z=300),
		hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
	)


def build_edge_trace(vertices: torch.Tensor) -> go.Scatter3d:
	"""Build a Plotly line trace outlining every triangle edge."""
	triangles = vertices.detach().cpu().numpy()

	# Draw triangle edges by inserting None separators between segments.
	edge_x: List[Optional[float]] = []
	edge_y: List[Optional[float]] = []
	edge_z: List[Optional[float]] = []

	for tri in triangles:
		p0, p1, p2 = tri
		edge_x.extend([p0[0], p1[0], None, p1[0], p2[0], None, p2[0], p0[0], None])
		edge_y.extend([p0[1], p1[1], None, p1[1], p2[1], None, p2[1], p0[1], None])
		edge_z.extend([p0[2], p1[2], None, p1[2], p2[2], None, p2[2], p0[2], None])

	return go.Scatter3d(
		x=edge_x,
		y=edge_y,
		z=edge_z,
		mode="lines",
		line=dict(color="rgba(35, 35, 35, 0.55)", width=1),
		hoverinfo="skip",
		showlegend=False,
	)


def flatten_intersection_points(points_list: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Flatten meshSlicing's per-plane `points_list` (list of (N_p, 3) tensors)
	into a single (N, 3) points array and a matching (N,)
	plane-id array, ready for plotting.
	"""
	all_points = []
	all_plane_ids = []
	for plane_index, points in enumerate(points_list):
		if points.numel() == 0:
			continue
		flat = points.detach().cpu().numpy()
		all_points.append(flat)
		all_plane_ids.append(np.full(len(flat), plane_index, dtype=np.int64))

	if not all_points:
		return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

	points = np.concatenate(all_points, axis=0)
	plane_ids = np.concatenate(all_plane_ids, axis=0)
	return points, plane_ids


def build_points_trace(points: np.ndarray, plane_ids: np.ndarray) -> go.Scatter3d:
	"""Build a Plotly marker trace for intersection points, colored by plane id."""
	return go.Scatter3d(
		x=points[:, 0],
		y=points[:, 1],
		z=points[:, 2],
		mode="markers",
		marker=dict(size=4, color=plane_ids, colorscale="Rainbow", opacity=0.95, showscale=False),
		name="Intersection points",
		showlegend=False,
		text=[f"plane {pid}" for pid in plane_ids],
		hovertemplate=(
			"Intersection point<br>%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
		),
	)


def build_figure(
	vertices: torch.Tensor,
	show_edges: bool = False,
	points_list: Optional[List[torch.Tensor]] = None,
	title: str = "Interactive 3D Model Viewer (Plotly)",
) -> go.Figure:
	"""Build a figure showing only the model mesh and (optionally) intersection points - no plane geometry."""
	figure = go.Figure()

	figure.add_trace(build_mesh_trace(vertices))

	if show_edges:
		figure.add_trace(build_edge_trace(vertices))

	if points_list:
		points, plane_ids = flatten_intersection_points(points_list)
		if len(points) > 0:
			figure.add_trace(build_points_trace(points, plane_ids))

	figure.update_layout(
		title=title,
		scene=dict(
			xaxis_title="X",
			yaxis_title="Y",
			zaxis_title="Z",
			aspectmode="data",
		),
		template="plotly_white",
		margin=dict(l=0, r=0, t=70, b=0),
	)

	return figure


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Visualize a 3D model (STL/OFF/OBJ/PLY) with optional slicing intersection points."
	)
	parser.add_argument("model_file", help="Path to the 3D model file (.stl, .off, .obj, .ply).")
	parser.add_argument("--edges", action="store_true", help="Draw triangle edges.")
	parser.add_argument(
		"--slice",
		action="store_true",
		help="Compute slicing-plane intersection points via meshSlicing and show them.",
	)
	parser.add_argument(
		"--plane-axis",
		type=float,
		nargs=3,
		default=None,
		metavar=("NX", "NY", "NZ"),
		help="Shared normal direction of the slicing planes. Defaults to (0, 0, 1).",
	)
	parser.add_argument(
		"--plane-interval",
		type=float,
		default=2.0,
		help="Gap between consecutive slicing planes.",
	)
	parser.add_argument("--output", default="model_viewer.html", help="Output HTML path.")
	args = parser.parse_args()

	model_path = Path(args.model_file).resolve()
	if not model_path.exists() or not model_path.is_file():
		raise SystemExit(f"Model file not found: {model_path}")

	vertices = loadModel(str(model_path))

	points_list = None
	if args.slice:
		plane_axis = tuple(args.plane_axis) if args.plane_axis else None
		_, points_list, _ = meshSlicing(vertices, plane_interval=args.plane_interval, plane_axis=plane_axis)

	figure = build_figure(
		vertices,
		show_edges=args.edges,
		points_list=points_list,
		title=f"Interactive 3D Model Viewer (Plotly)<br><sup>{model_path}</sup>",
	)

	output_path = Path(args.output).resolve()
	# Embed plotly.js directly in the HTML instead of loading it from a CDN.
	# Preview surfaces (Live Preview, Simple Browser, remote/sandboxed environments)
	# often have no internet access, so a CDN <script> tag silently fails to load
	# and the page renders blank with no visible error.
	figure.write_html(output_path, include_plotlyjs=True)
	print(f"Figure written to: {output_path}")
	print("Open this file in your local browser (e.g. right-click it in the VS Code")
	print("Explorer and choose 'Reveal in File Explorer', or open it with the")
	print("'Live Preview' / 'Open with Live Server' extension) instead of using")
	print("figure.show(), which relies on a fragile one-shot local HTTP server that")
	print("does not survive VS Code port forwarding.")


if __name__ == "__main__":
	main()

