import json
from pathlib import Path

_HPARAMS_PATH = Path(__file__).parent.parent / "hyperparameters.json"

def load_hyperparameters() -> dict:
    if not _HPARAMS_PATH.exists():
        raise FileNotFoundError(
            f"hyperparameters.json not found at {_HPARAMS_PATH}.\n"
            "Download the full model from Zenodo or run training first."
        )
    with open(_HPARAMS_PATH) as f:
        return json.load(f)

def save_hyperparameters(params: dict) -> None:
    with open(_HPARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"✓ hyperparameters.json updated at {_HPARAMS_PATH}")
