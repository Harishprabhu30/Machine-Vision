import subprocess
import sys
import os
sys.path.append(os.path.abspath("./src"))  # Add src/ to Python path
from src.logger import logger

logger.info("Starting full ML pipeline...")

# Run each stage sequentially
subprocess.run(["python", "scripts/data_ingestion.py"])
subprocess.run(["python", "scripts/visualizations.py"])
# subprocess.run(["python", "scripts/traditional_feature_engineering.py"])
# subprocess.run(["python", "scripts/dnn_feature_engineering.py"])
# # subprocess.run(["python", "scripts/train.py"])
# # subprocess.run(["python", "scripts/evaluate.py"])

logger.info("ML pipeline completed.")