import sqlite3
import yaml
from pathlib import Path
from src.utils.logger import logger

def get_db_path():
    try:
        with open("configs/tools.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("database_path", "data/usda.db")
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using default path.")
        return "data/usda.db"

def get_connection():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None
