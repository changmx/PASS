"""Schema definitions for PASS simulation parameters.

All parameter models use pydantic v2 BaseModel with alias fields.
The alias is the JSON key consumed by the PASS engine (case-insensitive
after convert_keys_to_lower). The Python attribute name is clean.

Models:
    MainConfig       — global simulation parameters
    BunchConfig      — per-bunch injection parameters
    OffsetConfig     — injection offset (x or y)
    TwissPoint       — twiss transport point
    ElementBase      — base for all magnet/element types
    DriftElement ... — specific element types
    StatMonitor ...  — monitor types
    Sequence         — ordered container + sort
"""
