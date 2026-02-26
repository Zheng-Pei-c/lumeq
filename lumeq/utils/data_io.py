import json
import warnings
from pathlib import Path

from lumeq import np

def _json_default(obj):
    """Default JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def _to_numpy(obj, path=()):
    """Recursively convert lists to numpy arrays."""
    if isinstance(obj, dict):
        return {k: _to_numpy(v, path + (k,)) for k, v in obj.items()}

    if isinstance(obj, list):
        converted = [_to_numpy(v, path) for v in obj]
        try:
            array = np.array(converted)
        except ValueError:
            # Keep ragged or mixed-content lists as Python lists.
            return converted
        if array.dtype == object:
            return converted
        return array

    return obj


def save_json(file_name, data, indent=4):
    """Save data to a JSON file."""
    if not file_name.endswith('.json'):
        file_name += '.json'
    if Path(file_name).is_file():
        warnings.warn(f"File '{file_name}' already exists and will be overwritten.", UserWarning)

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=indent, default=_json_default)


def load_json(file_name, to_numpy=False):
    """Load data from a JSON file."""
    if not file_name.endswith('.json'):
        file_name += '.json'
    if not Path(file_name).is_file():
        warnings.warn(f"File '{file_name}' does not exist.")
        return None

    with open(file_name, 'r') as f:
        data = json.load(f)

    if to_numpy:
        data = _to_numpy(data)
    return data
