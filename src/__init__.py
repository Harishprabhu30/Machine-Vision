import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()