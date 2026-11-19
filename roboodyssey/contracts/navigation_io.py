"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

import struct

import json

from pathlib import Path

import zipfile

def _npy(value: object, dtype: str, shape: tuple[int, ...]) -> bytes:
    """Write a small standard NumPy .npy v1 array without importing NumPy."""
    flat = value if isinstance(value, list) else [value]
    if shape and len(shape) == 2:
        flat = [item for row in flat for item in row]
    if dtype == "|u1":
        raw = bytes(int(item) & 255 for item in flat)
    elif dtype == "<i4":
        raw = b"".join(struct.pack("<i", int(item)) for item in flat)
    elif dtype == "<f8":
        raw = b"".join(struct.pack("<d", float(item)) for item in flat)
    elif dtype.startswith("<U"):
        width = int(dtype[2:])
        raw = b"".join(
            str(item).encode("utf-32-le")[: width * 4].ljust(width * 4, b"\0")
            for item in flat
        )
    else:
        raise ValueError(f"unsupported NPZ dtype: {dtype}")
    shape_text = "()" if not shape else "(" + ", ".join(str(item) for item in shape) + ("," if len(shape) == 1 else "") + ")"
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (dtype, shape_text)
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * (padding - 1) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + raw

def write_navigation_npz(path: str | Path, navigation: dict) -> None:
    """Write a navigation dictionary using the repository's deterministic NPZ layout."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height = len(navigation["static_occupancy"])
    width = len(navigation["static_occupancy"][0])
    static_base = navigation.get("static_base_occupancy", navigation["static_occupancy"])
    arrays = {
        "format_version": _npy(1, "<i4", ()),
        "scene_id": _npy(navigation["scene_id"], "<U" + str(max(1, len(navigation["scene_id"]))), ()),
        "resolution": _npy(navigation["resolution"], "<f8", ()),
        "origin_xy": _npy(navigation["origin_xy"], "<f8", (2,)),
        "floor_height": _npy(navigation["floor_height"], "<f8", ()),
        "robot_radius": _npy(navigation["robot_radius"], "<f8", ()),
        "connectivity": _npy(navigation["connectivity"], "|u1", ()),
        "static_occupancy": _npy(navigation["static_occupancy"], "|u1", (height, width)),
        "static_base_occupancy": _npy(static_base, "|u1", (height, width)),
        "dynamic_occupancy": _npy(navigation["dynamic_occupancy"], "|u1", (height, width)),
        "room_labels": _npy(navigation["room_labels"], "<i4", (height, width)),
        "room_ids": _npy(navigation["room_ids"], "<U" + str(max([1] + [len(str(item)) for item in navigation["room_ids"]])), (len(navigation["room_ids"]),)),
    }
    if navigation.get('environment_setup'):
        setup = json.dumps(navigation['environment_setup'], sort_keys=True, separators=(',', ':'))
        arrays['environment_setup_json'] = _npy(setup, '<U' + str(len(setup)), ())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)

def read_navigation_npz(path: str | Path) -> dict:
    """Read numeric/string arrays without allowing object pickle payloads."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].tolist() for name in archive.files}
    if 'environment_setup_json' in result:
        result['environment_setup'] = json.loads(result.pop('environment_setup_json'))
    return result
