import numpy as np
import json
import tfs
from datetime import datetime


def convert_keys_to_lower(obj):
    # Recursively convert all keys in the dictionary to lowercase
    if isinstance(obj, dict):
        return {k.lower(): convert_keys_to_lower(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_lower(item) for item in obj]
    else:
        return obj


def get_current_time():
    """
    Get current time in specified format

    Returns:
        timestamp: str
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
