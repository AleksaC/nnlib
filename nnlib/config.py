"""nnlib configuration

Configuration is stored in a json file in a hidden directory in home folder.
When the model is imported the configuration is loaded from the file if it
exists otherwise the defaults are set and the file is created. If you want to
change the config in runtime all you have to do is monkey-patch the parameter
you want to change. If you want to make the changes permanent all you have to
do is call the save method from this module after assigning the new values.
"""

import json
import os


_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".nnlib")
_DEFAULT_CONFIG = {
    "base_dir": _CONFIG_DIR,
    "epsilon": 1e-8,
    "backend": 'numpy'
}


def _load_config():
    """Loads configuration from config.json.

    Loads configuration from config.json located in ~/.nnlib
    directory. If the file doesn't exist it creates the file with
    the default configuration.

    Returns:
        config: dictionary of configuration values containing
        epsilon and base_dir
    """
    try:
        with open(os.path.join(_CONFIG_DIR, "config.json")) as f:
            config = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(_CONFIG_DIR):
            os.makedirs(_CONFIG_DIR)

        config = _DEFAULT_CONFIG

        with open(os.path.join(_CONFIG_DIR, "config.json"), "w") as f:
            json.dump(config, f)

    return config


def save():
    """Saves configuration modifications.

    Configuration is performed by monkey-patching the desired variable in this
    module. Note that modifying the parameters from this module either temporarily
    or permanently should be performed BEFORE importing other nnlib
    modules as they may have already set some of their internal parameters to
    the values from config. The usage of this function should be clear from
    the following example.

    Examples:
        >>>from nnlib import config
        >>>config.epsilon = 1e-10
        >>>config.save()
    """
    with open(os.path.join(_CONFIG_DIR, "config.json"), "w") as f:
        json.dump({
            "base_dir": base_dir,
            "epsilon": epsilon,
            "backend": backend
        }, f)


_config = _load_config()
base_dir = _config.get("base_dir", _DEFAULT_CONFIG.get("base_dir"))
epsilon  = _config.get("epsilon", _DEFAULT_CONFIG.get("epsilon"))
backend  = _config.get("backend", _DEFAULT_CONFIG.get("backend"))
