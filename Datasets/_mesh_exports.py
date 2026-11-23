"""Public mesh helpers shared by the Datasets and ThreeDUtils namespaces."""

MESH_MODULES = {
    "loadSTL": "FileLoading",
    "loadOFF": "FileLoading",
    "loadOBJ": "FileLoading",
    "loadPLY": "FileLoading",
    "meshSlicing": "ModelSlicing",
    "meshSlicing_vectorized": "ModelSlicing",
    "normalizeMesh": "ProjectionUtils",
    "denormalizePoints": "ProjectionUtils",
    "sampleAxis": "ProjectionUtils",
    "buildPlaneBasis": "ProjectionUtils",
    "projectPointsTo2D": "ProjectionUtils",
    "reconstructPoints3D": "ProjectionUtils",
    "computeSDFImages": "SDFUtils",
    "loadModel": "ViewModel",
    "build_figure": "ViewModel",
}
