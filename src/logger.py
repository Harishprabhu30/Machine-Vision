import logging
import yaml
from pathlib import Path

# Load Config file
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Loagging Setup
LOGS_PATH = Path(config["paths"]['logs'])
LOGS_PATH.mkdir(parents = True, exist_ok = True)

logging.basicConfig(
    filename = LOGS_PATH / 'pipeline.log',
    level = getattr(logging, config['logging']['level']),
    format = config['logging']['format']
)

logger = logging.getLogger(__name__)