import ast
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = ["01_prepare_data", "02_train_prism", "03_train_baselines", "04_train_cnn", "05_evaluate",
            "06_inverse_map_laplace", "07_inverse_mcmc", "08_optuna"]
INSTALL = [
    'PRISM_REF = "main"',
    "!pip freeze | grep -E '^(jax|jaxlib|jax-cuda|flax|orbax|optax|numpy|scipy)' > /tmp/kaggle-pins.txt",
    '!pip install -q -c /tmp/kaggle-pins.txt "git+https://github.com/Aneeshshastri/PRISM@{PRISM_REF}" blackjax',
]


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_all_notebooks_present():
    assert sorted(p.stem for p in NB_DIR.glob("*.ipynb")) == EXPECTED


def test_notebooks_valid_and_thin():
    for path in sorted(NB_DIR.glob("*.ipynb")):
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4
        code = _code(nb)
        assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), path.name
        joined = "\n".join(code)
        for line in INSTALL:
            assert line in joined, (path.name, line)
        assert "check_env(" in joined, path.name
        if "checkpoint_inputs" in joined:  # nested Kaggle mounts need a recursive glob
            assert "/kaggle/input/**/prism_artifacts/checkpoints" in joined, path.name
        assert "from prism import stages" in joined or "prism.stages" in joined, path.name
        for src in code:
            stripped = "\n".join("" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
            ast.parse(stripped)
        assert len(joined.splitlines()) < 80, f"{path.name} is not thin"
