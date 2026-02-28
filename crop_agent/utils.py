import yaml
from pathlib import Path

# Root of the project (two levels up from this file)
_CONFIG_DIR = Path(__file__).parent.parent / "config"

# YAML config file paths
AGENT_CONFIG = _CONFIG_DIR / "agent.yaml"
DATABASE_CONFIG = _CONFIG_DIR / "database.yaml"
DEPLOYMENT_CONFIG = _CONFIG_DIR / "deployment.yaml"
MODEL_CONFIG = _CONFIG_DIR / "model.yaml"
SENSORS_CONFIG = _CONFIG_DIR / "sensors.yaml"
SYSTEM_CONFIG = _CONFIG_DIR / "system_config.yaml"


def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Usage:
        config = load_yaml(SYSTEM_CONFIG)
        config = load_yaml("config/agent.yaml")
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
