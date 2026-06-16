import numpy as np
import json
import tfs


def convert_keys_to_lower(obj):
    # Recursively convert all keys in the dictionary to lowercase
    if isinstance(obj, dict):
        return {k.lower(): convert_keys_to_lower(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_lower(item) for item in obj]
    else:
        return obj
