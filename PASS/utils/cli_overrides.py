"""Schema-aware, per-run JSON overrides without modifying source files."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter


@dataclass(frozen=True)
class InputOverride:
    beam: int
    path: tuple
    value: str


def parameter_name(key):
    """Only fixed schema fields are normalized; user-defined names stay exact."""
    return re.sub(r'\s+', '-', re.sub(r'\([^()]*\)', '', key).strip()).casefold()


def _parse_path(path):
    result = []
    position = 0
    while position < len(path):
        match = re.match(r'\[([^\[\]]+)\]|([^.[\]]+)', path[position:])
        if match is None:
            raise ValueError(f'Invalid override path: {path!r}')
        result.append((match.group(1) or match.group(2), match.group(1) is not None))
        position += match.end()
        if position < len(path) and path[position] == '.':
            position += 1
            if position == len(path) or path[position] in '.[':
                raise ValueError(f'Invalid override path: {path!r}')
        elif position < len(path) and path[position] != '[':
            raise ValueError(f'Invalid override path: {path!r}')
    if not result:
        raise ValueError('Override path must not be empty')
    return result


def extract_overrides(argv, known_options):
    """Separate arbitrary --path values before argparse consumes positionals."""
    remaining, overrides = [], []
    aliases = {
        '--output': 'output-directory',
        '--output-directory': 'output-directory',
        '--output-subdirectory': 'task-name',
        '--output-name': 'task-name'
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == '--':
            remaining.extend(argv[index:])
            break
        option, equals, value = token.partition('=')
        if option == '--set':
            if equals or index + 2 >= len(argv):
                raise ValueError('--set requires a JSON key-path array and a value')
            path = json.loads(argv[index + 1])
            if not isinstance(path, list) or not path or any(type(part) not in (str, int) for part in path):
                raise ValueError('--set path must be a nonempty JSON array of keys and indices')
            parts = [(part, True) for part in path]
            value = argv[index + 2]
            index += 3
        elif option.startswith('--') and (option not in known_options or option in aliases or option == '--task-name'):
            parts = _parse_path(aliases.get(option, option[2:]))
            if not equals:
                if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
                    raise ValueError(f'{option} requires a value; use {option}=VALUE')
                value = argv[index + 1]
                index += 1
            index += 1
        else:
            remaining.append(token)
            index += 1
            action = known_options.get(option)
            if action is not None and action.nargs != 0 and not equals and index < len(argv):
                remaining.append(argv[index])
                index += 1
            continue
        beam = 0
        if parts[0][0] in ('beam0', 'beam1'):
            beam = int(parts.pop(0)[0][-1])
        if not parts:
            raise ValueError('A beam prefix must be followed by a field path')
        overrides.append(InputOverride(beam, tuple(parts), value))
    return remaining, overrides


def _select_annotation(annotation, value):
    if get_origin(annotation) is Annotated:
        return _select_annotation(get_args(annotation)[0], value)
    if get_origin(annotation) in (Union, UnionType):
        options = [item for item in get_args(annotation) if item is not type(None)]
        if len(options) == 1:
            return _select_annotation(options[0], value)
        if isinstance(value, dict):
            kind = next((item for key, item in value.items() if key.casefold() == 'kind'), None)
            models = [item for item in options if isinstance(item, type) and issubclass(item, BaseModel)]
            matched = [item for item in models if 'kind' in item.model_fields and kind in get_args(item.model_fields['kind'].annotation)]
            if len(matched) == 1:
                return matched[0]
        if isinstance(value, list):
            return next((item for item in options if get_origin(item) is list), annotation)
    return annotation


def _command_model(value):
    from PASS.validation.rules import MODELS

    kind = next((item for key, item in value.items() if key.casefold() == 'command'), '')
    return next((model for name, model in MODELS.items() if name.casefold() == str(kind).casefold()), None)


def _fields(annotation, value, path):
    from PASS.validation.rules import (MainConfig, InjectionItem, BunchConfig, SlicerItem, SpaceChargeConfig, WakeFieldConfig, BeamBeamConfig,
                                       ElectronCloudConfig, IBSConfig)

    annotation = _select_annotation(annotation, value)
    if path == ('Sequence', ):
        return {key: _command_model(item) for key, item in value.items() if isinstance(item, dict)}, True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields = {
            field.alias or name: Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
            for name, field in annotation.model_fields.items()
        }
        if annotation is MainConfig:
            fields.update({
                'Sequence': dict,
                'Space charge': SpaceChargeConfig,
                'Wake field': WakeFieldConfig,
                'Beam beam': BeamBeamConfig,
                'Electron cloud': ElectronCloudConfig,
                'Intrabeam scattering': IBSConfig
            })
        if annotation is InjectionItem:
            fields.pop('bunches', None)
            fields.update({key: BunchConfig for key in value if re.fullmatch(r'bunch\d+', key)})
        if annotation is SlicerItem:
            fields['Explicit'] = {'z min': float, 'z max': float}
        return fields, False
    if isinstance(annotation, dict):
        return annotation, False
    child = get_args(annotation)[1] if get_origin(annotation) is dict and len(get_args(annotation)) == 2 else None
    return {key: child for key in value}, True


def _resolve_key(value, annotation, path, token, exact):
    fields, named = _fields(annotation, value, path)
    if named or exact:
        matches = [key for key in fields if key == token]
    else:
        matches = [key for key in fields if parameter_name(key) == parameter_name(token)]
    if len(matches) != 1:
        raise ValueError(f'Unknown or ambiguous field {token!r} beneath {path}; matches: {matches}')
    canonical = matches[0]
    existing = [key for key in value if key == canonical or (not named and key.casefold() == canonical.casefold())]
    if len(existing) > 1:
        raise ValueError(f'Duplicate JSON keys for {canonical!r}')
    return existing[0] if existing else canonical, canonical, fields[canonical]


def _decode_value(raw, annotation, old_value, path):

    def read_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key in override: {key!r}')
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=read_object)
    except json.JSONDecodeError:
        value = raw
    if path == ('Device Id', ) and type(value) is int:
        value = [value]
    if len(path) == 2 and path[0] == 'Sequence' and isinstance(value, dict):
        annotation = _command_model(value)
    if annotation is not None and not isinstance(annotation, dict):
        adapter = TypeAdapter(annotation)
        try:
            adapter.validate_python(value, strict=True)
        except ValueError:
            # Numeric-looking names are strings when the schema accepts a string.
            adapter.validate_python(raw, strict=True)
            value = raw
    elif old_value is not None and type(value) is not type(old_value):
        if isinstance(old_value, str):
            value = raw
        else:
            raise ValueError(f'Expected {type(old_value).__name__}, received {type(value).__name__}')
    return value


def _check_structure(old, new, annotation, path):
    """Allow parameter replacement, but not silently changing named objects."""
    if isinstance(new, dict):
        annotation = _select_annotation(annotation, new)
        if len(path) == 2 and path[0] == 'Sequence':
            annotation = _command_model(new)
        old = old if isinstance(old, dict) else {}
        fields, named = _fields(annotation, old, path)
        if named and set(old) != set(new):
            raise ValueError(f'Cannot add or remove named objects at {path}')
        for key, item in new.items():
            old_key, canonical, child = _resolve_key(old, annotation, path, key, False)
            _check_structure(old.get(old_key), item, child, (*path, canonical))
    elif isinstance(new, list):
        selected = _select_annotation(annotation, new)
        child = get_args(selected)[0] if get_origin(selected) is list else None
        for index, item in enumerate(new):
            previous = old[index] if isinstance(old, list) and index < len(old) else None
            _check_structure(previous, item, child, (*path, index))


def apply_overrides(documents, overrides):
    """Return copied documents and explicit shared output settings, if supplied."""
    from PASS.para.schema.main import MainConfig
    from PASS.utils.input_snapshot import canonicalize_task_name, file_references

    documents = [(label, canonicalize_task_name(deepcopy(data)), base) for label, data, base in documents]
    used, shared = [], {}
    for override in overrides:
        if override.beam >= len(documents):
            raise ValueError(f'beam{override.beam} is not present in this run')
        value = documents[override.beam][1]
        annotation, path = MainConfig, ()
        for position, (token, exact) in enumerate(override.path):
            final = position == len(override.path) - 1
            if isinstance(value, list):
                if not re.fullmatch(r'\d+', str(token)) or int(token) >= len(value):
                    raise ValueError(f'Array index {token!r} out of range at {path}')
                key = canonical = int(token)
                selected = _select_annotation(annotation, value)
                child = get_args(selected)[0] if get_origin(selected) is list else None
            elif isinstance(value, dict):
                key, canonical, child = _resolve_key(value, annotation, path, token, exact)
            else:
                raise ValueError(f'Cannot traverse scalar at {path}')
            path = (*path, canonical)
            if final:
                identity = (override.beam, *path)
                if path[0] in ('Task name', 'Output directory'):
                    if override.beam != 0:
                        raise ValueError(f'{path[0]} is shared; specify it without a beam1 prefix')
                if any(identity[:len(previous)] == previous or previous[:len(identity)] == identity for previous in used):
                    raise ValueError(f'Duplicate or overlapping overrides for {path}')
                used.append(identity)
                old = value[key] if isinstance(value, list) else value.get(key)
                new = _decode_value(override.value, child, old, path)
                _check_structure(old, new, child, path)
                wrapper = {canonical: new}
                # Resolve only command-line supplied file paths against the caller.
                for parent, field, _ in file_references(wrapper):
                    parent[field] = str(Path(parent[field]).expanduser().resolve())
                new = wrapper[canonical]
                if path == ('Output directory', ):
                    if not new.strip():
                        raise ValueError('Output directory must not be empty')
                    new = str(Path(new).expanduser().resolve())
                value[key] = new
                if path[0] in ('Task name', 'Output directory'):
                    shared[path[0]] = new
            else:
                if isinstance(value, dict) and (key not in value or value[key] is None):
                    selected = _select_annotation(child, None)
                    if not (isinstance(selected, dict) or isinstance(selected, type) and issubclass(selected, BaseModel)):
                        raise ValueError(f'Cannot traverse missing container at {path}; replace the whole value')
                    value[key] = {}
                value = value[key]
                annotation = child
    return documents, shared
