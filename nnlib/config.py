"""nnlib configuration"""
import json
import os


CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".nnlib")


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
        with open(os.path.join(CONFIG_DIR, "config.json")) as f:
            config = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)

        config = {
            "base_dir": CONFIG_DIR,
            "epsilon": 1e-8
        }
        with open(os.path.join(CONFIG_DIR, "config.json"), "w") as f:
            json.dump(config, f)

    return config


def save():
    """Saves configuration modifications.

    Configuration should be modified by importing this module
    and then changing the value of the variable we want to
    modify and calling this method to make changes permanent.
    This should be done before importing any other nnlib modules.

    Examples:
        >>>from nnlib import config
        >>>config.epsilon = 1e-10
        >>>config.save()
    """
    with open(os.path.join(CONFIG_DIR, "config.json"), "w") as f:
        json.dump({
            "base_dir": base_dir,
            "epsilon": epsilon
        }, f)


_config = _load_config()
base_dir = _config["base_dir"]
epsilon = _config["epsilon"]
