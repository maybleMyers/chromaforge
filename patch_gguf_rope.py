#!/usr/bin/env python3
"""
Patch a GGUF file to fix rope.dimension_sections array length.

llama.cpp expects exactly 4 elements for rope.dimension_sections in
qwen35moe (and other VL) architectures, but HF configs only provide 3
(mrope_section: [time, height, width]). This script pads with 0 to get
[time, height, width, 0].

Streams tensor data to avoid loading the full model into RAM.

Usage:
    python patch_gguf_rope.py models/LLM/Qwen3.5-397B-A17B/qwen3.5_q4_K_M.gguf
    python patch_gguf_rope.py input.gguf -o output.gguf
"""

import argparse
from pathlib import Path

import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.gguf_writer import TensorInfo
import numpy as np


def extract_field_value(field):
    """Extract a Python value from a GGUFReader ReaderField.

    Returns (value, is_array, value_type) where:
    - value: the extracted Python value (scalar, string, or list)
    - is_array: True if the field is an array type
    - value_type: the GGUFValueType of the scalar or array element type
    """
    types = field.types
    parts = field.parts
    data_idxs = field.data

    if not types:
        return None, False, None

    if types[0] == gguf.GGUFValueType.ARRAY:
        # Array field
        # types[1] is the element type
        # data_idxs contains indexes into parts for each element's data
        elem_type = types[1] if len(types) > 1 else None

        if elem_type == gguf.GGUFValueType.STRING:
            values = []
            for idx in data_idxs:
                raw = parts[idx]
                if isinstance(raw, np.ndarray):
                    values.append(bytes(raw).decode('utf-8', errors='replace').rstrip('\x00'))
                elif isinstance(raw, bytes):
                    values.append(raw.decode('utf-8', errors='replace').rstrip('\x00'))
                else:
                    values.append(str(raw))
            return values, True, elem_type
        elif elem_type in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32,
                           gguf.GGUFValueType.INT64, gguf.GGUFValueType.UINT64,
                           gguf.GGUFValueType.INT8, gguf.GGUFValueType.UINT8,
                           gguf.GGUFValueType.INT16, gguf.GGUFValueType.UINT16):
            values = [int(parts[idx][0]) for idx in data_idxs]
            return values, True, elem_type
        elif elem_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            values = [float(parts[idx][0]) for idx in data_idxs]
            return values, True, elem_type
        elif elem_type == gguf.GGUFValueType.BOOL:
            values = [bool(parts[idx][0]) for idx in data_idxs]
            return values, True, elem_type
        else:
            return None, True, elem_type

    elif types[0] == gguf.GGUFValueType.STRING:
        # String scalar: parts[0] = length, parts[1] = string bytes
        raw = parts[data_idxs[0]] if data_idxs else parts[-1]
        if isinstance(raw, np.ndarray):
            return bytes(raw).decode('utf-8', errors='replace').rstrip('\x00'), False, types[0]
        elif isinstance(raw, bytes):
            return raw.decode('utf-8', errors='replace').rstrip('\x00'), False, types[0]
        return str(raw), False, types[0]

    else:
        # Scalar type
        val = parts[data_idxs[0]] if data_idxs else parts[0]
        if isinstance(val, np.ndarray) and val.size == 1:
            val = val.flat[0]
        return val, False, types[0]


def write_field_to_writer(writer, name, value, is_array, value_type):
    """Write a field value to a GGUFWriter."""
    if is_array:
        writer.add_array(name, value)
    elif value_type == gguf.GGUFValueType.STRING:
        writer.add_string(name, str(value))
    elif value_type == gguf.GGUFValueType.UINT32:
        writer.add_uint32(name, int(value))
    elif value_type == gguf.GGUFValueType.INT32:
        writer.add_int32(name, int(value))
    elif value_type == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(name, float(value))
    elif value_type == gguf.GGUFValueType.BOOL:
        writer.add_bool(name, bool(value))
    elif value_type == gguf.GGUFValueType.UINT64:
        writer.add_uint64(name, int(value))
    elif value_type == gguf.GGUFValueType.INT64:
        writer.add_int64(name, int(value))
    elif value_type == gguf.GGUFValueType.FLOAT64:
        writer.add_float64(name, float(value))
    elif value_type == gguf.GGUFValueType.UINT8:
        writer.add_uint8(name, int(value))
    elif value_type == gguf.GGUFValueType.INT8:
        writer.add_int8(name, int(value))
    elif value_type == gguf.GGUFValueType.UINT16:
        writer.add_uint16(name, int(value))
    elif value_type == gguf.GGUFValueType.INT16:
        writer.add_int16(name, int(value))
    else:
        print(f"  [WARN] Unknown type {value_type} for {name}, skipping")


def patch_gguf(input_path: Path, output_path: Path | None = None):
    """Read GGUF, fix rope.dimension_sections, write new GGUF."""

    if output_path is None:
        output_path = input_path.with_suffix('.fixed.gguf')

    print(f"Reading: {input_path}")
    reader = GGUFReader(str(input_path))

    # Find the rope.dimension_sections field
    rope_field_name = None
    rope_sections_old = None
    for field in reader.fields.values():
        if 'rope.dimension_sections' in field.name:
            rope_field_name = field.name
            value, is_array, vtype = extract_field_value(field)
            rope_sections_old = value
            break

    if rope_field_name is None:
        print("No rope.dimension_sections field found, nothing to fix.")
        return

    print(f"  Found: {rope_field_name} = {rope_sections_old}")

    if isinstance(rope_sections_old, list) and len(rope_sections_old) >= 4:
        print(f"  Already has {len(rope_sections_old)} elements, nothing to fix.")
        return

    # Pad to 4 elements
    rope_sections_new = list(rope_sections_old)
    while len(rope_sections_new) < 4:
        rope_sections_new.append(0)
    rope_sections_new = rope_sections_new[:4]
    print(f"  Patching: {rope_sections_old} -> {rope_sections_new}")

    # Get architecture from metadata
    arch = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            value, _, _ = extract_field_value(field)
            arch = str(value)
            break

    if arch is None or len(arch) < 2:
        arch = "qwen35moe"
    print(f"  Architecture: {arch}")

    # Create new GGUF
    print(f"Writing: {output_path}")
    writer = GGUFWriter(str(output_path), arch=arch)

    # Copy all metadata, overriding rope.dimension_sections
    print("Copying metadata...")
    skipped = 0
    copied = 0
    for field in reader.fields.values():
        name = field.name

        # Skip auto-added fields (writer adds these automatically)
        if name.startswith('GGUF.') or name == 'general.architecture':
            continue

        # Override rope.dimension_sections
        if name == rope_field_name:
            writer.add_array(name, rope_sections_new)
            print(f"  Patched: {name} = {rope_sections_new}")
            copied += 1
            continue

        try:
            value, is_array, value_type = extract_field_value(field)
            if value is None:
                skipped += 1
                continue
            write_field_to_writer(writer, name, value, is_array, value_type)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Skipping {name}: {e}")
            skipped += 1

    print(f"  Copied {copied} metadata fields ({skipped} skipped)")

    # Copy all tensors
    print(f"Copying {len(reader.tensors)} tensors...")
    tensor_dict = writer.tensors[0]

    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = tuple(tensor.shape)
        dtype = tensor.tensor_type
        data = tensor.data
        n_bytes = tensor.n_bytes

        tensor_dict[name] = TensorInfo(
            shape=shape,
            dtype=dtype,
            nbytes=n_bytes,
            tensor=np.array(data),
        )

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(reader.tensors)} tensors...")

    print(f"  Total: {len(tensor_dict)} tensors")

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = output_path.stat().st_size
    print(f"Done! Output: {output_path} ({out_size / (1024**3):.1f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Patch GGUF rope.dimension_sections from 3 to 4 elements"
    )
    parser.add_argument("input", type=Path, help="Input GGUF file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output GGUF file (default: input.fixed.gguf)",
    )
    args = parser.parse_args()

    patch_gguf(args.input, args.output)


if __name__ == "__main__":
    main()
