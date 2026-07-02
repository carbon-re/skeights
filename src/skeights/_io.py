"""Save and load fitted sklearn models to safetensors + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import safetensors.numpy

from skeights._core import SklearnModel, SklearnScaler

_REGISTRY: dict[str, type] = {
    "SklearnModel": SklearnModel,
    "SklearnScaler": SklearnScaler,
}


def save(
    model: SklearnModel | SklearnScaler,
    arrays_path: str | Path,
    state_path: str | Path,
) -> None:
    """Serialize a fitted model to safetensors + JSON files.

    Args:
        model: A fitted SklearnModel or SklearnScaler.
        arrays_path: Destination path for the safetensors weight file.
        state_path: Destination path for the JSON state file.
    """
    arrays_path = Path(arrays_path)
    state_path = Path(state_path)
    safetensors.numpy.save_file(model.get_arrays(), str(arrays_path))
    state_path.write_text(json.dumps(model.get_state(), indent=2))


def load(
    arrays_path: str | Path,
    state_path: str | Path,
) -> SklearnModel | SklearnScaler:
    """Reconstruct a fitted model from safetensors + JSON files.

    Args:
        arrays_path: Path to the safetensors weight file.
        state_path: Path to the JSON state file.

    Returns:
        A fully reconstructed, ready-to-predict model instance.
    """
    arrays_path = Path(arrays_path)
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    arrays = dict(safetensors.numpy.load_file(str(arrays_path)))
    type_key = state.get("type", "SklearnModel")
    cls = _REGISTRY[type_key]
    return cls.from_state(state, arrays)
