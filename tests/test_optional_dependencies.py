"""Exercise normal entry points in a fresh interpreter with feature packages absent."""
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent


def test_training_and_mesh_helpers_without_extras(tmp_path):
    script = dedent("""\
        import sys
        from pathlib import Path

        for name in ("stl", "plotly", "calflops", "aiohttp", "tensorboard"):
            sys.modules[name] = None

        import torch
        import main
        import distributed_main
        import Datasets
        from JimmyExperiment import JimmyExperiment

        assert "Datasets.ThreeDUtils" not in sys.modules
        assert "Training.ExperimentManager.server" not in sys.modules
        model = JimmyExperiment("optional dependencies").model_cfg.build()
        model.initialize()
        loss, output = model.trainStep({"input": torch.rand(2, 1, 28, 28), "target": torch.tensor([0, 1])})
        assert output["output"].shape == (2, 10)
        assert 0 <= loss["Train/Main"] < float("inf")

        from Datasets import *
        from Datasets.ThreeDUtils import loadModel
        assert Datasets.loadModel is loadModel
        path = Path(sys.argv[1]) / "triangle.obj"
        path.write_text("v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3\\n")
        vertices = loadModel(str(path))
        assert normalizeMesh(vertices)[0].shape == (1, 3, 3)

        for func, arg, package in [(loadModel, "sample.stl", "numpy-stl"), (build_figure, vertices, "plotly")]:
            try:
                func(arg)
            except ModuleNotFoundError as error:
                assert f"pip install {package}" in str(error), error
            else:
                raise AssertionError(f"Expected an install hint for {package}")
    """)
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        check=True,
    )
