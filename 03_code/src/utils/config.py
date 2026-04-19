"""
Config loader utility.

Owner: Sahitya
"""

import yaml


class DotDict(dict):
    """Dict subclass that supports attribute-style access (dot notation).

    Enables config.training.lr instead of config['training']['lr'].
    """

    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, val):
        self[key] = val

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def load_config(config_path):
    """Load YAML config file and return as DotDict for attribute-style access."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return DotDict(config)
