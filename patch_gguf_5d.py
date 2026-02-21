#!/usr/bin/env python3
"""
Patch a GGUF file to fix 5D vision tensors by reshaping to 4D.

This modifies the tensor shape metadata in-place without touching the data.
[out, temporal, in_ch, h, w] -> [out, temporal*in_ch, h, w]
"""

import argparse
import struct
import sys
from pathlib import Path

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
import numpy as np


def patch_gguf(input_path: Path, output_path: Path | None = None):
    """Read GGUF, fix 5D tensors, write new GGUF."""

    if output_path is None:
        output_path = input_path.with_suffix('.fixed.gguf')

    print(f"Reading: {input_path}")
    reader = GGUFReader(str(input_path))

    # Find tensors that need fixing
    tensors_to_fix = []
    for tensor in reader.tensors:
        if len(tensor.shape) > 4:
            tensors_to_fix.append(tensor.name)
            print(f"  Found 5D tensor: {tensor.name} shape={list(tensor.shape)}")

    if not tensors_to_fix:
        print("No 5D tensors found, nothing to fix.")
        return

    # Get architecture from metadata
    arch = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch = str(field.parts[-1][0], 'utf-8')
            break

    if arch is None:
        arch = "qwen35moe"
    print(f"  Architecture: {arch}")

    # Create new GGUF with fixed shapes
    print(f"Writing: {output_path}")
    writer = GGUFWriter(str(output_path), arch=arch)

    # Copy all metadata
    for field in reader.fields.values():
        if field.name.startswith('general.') or field.name == 'GGUF.version':
            continue  # Skip auto-added fields

        # Get raw value and type
        name = field.name
        parts = field.parts
        types = field.types

        # Skip complex nested fields for now, copy simple ones
        if len(parts) == 1 and len(types) == 1:
            val = parts[0]
            t = types[0]

            if t == gguf.GGUFValueType.STRING:
                writer.add_string(name, str(val[0], 'utf-8') if isinstance(val[0], bytes) else str(val[0]))
            elif t == gguf.GGUFValueType.UINT32:
                writer.add_uint32(name, int(val[0]))
            elif t == gguf.GGUFValueType.INT32:
                writer.add_int32(name, int(val[0]))
            elif t == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(name, float(val[0]))
            elif t == gguf.GGUFValueType.BOOL:
                writer.add_bool(name, bool(val[0]))
            elif t == gguf.GGUFValueType.UINT64:
                writer.add_uint64(name, int(val[0]))
            elif t == gguf.GGUFValueType.INT64:
                writer.add_int64(name, int(val[0]))
            elif t == gguf.GGUFValueType.FLOAT64:
                writer.add_float64(name, float(val[0]))
            elif t == gguf.GGUFValueType.UINT8:
                writer.add_uint8(name, int(val[0]))
            elif t == gguf.GGUFValueType.INT8:
                writer.add_int8(name, int(val[0]))
            elif t == gguf.GGUFValueType.UINT16:
                writer.add_uint16(name, int(val[0]))
            elif t == gguf.GGUFValueType.INT16:
                writer.add_int16(name, int(val[0]))
        elif len(types) >= 2 and types[0] == gguf.GGUFValueType.ARRAY:
            # Array field
            arr_type = types[1]
            arr_data = parts[-1]

            if arr_type == gguf.GGUFValueType.STRING:
                str_list = [str(s, 'utf-8') if isinstance(s, bytes) else str(s) for s in arr_data]
                writer.add_array(name, str_list)
            elif arr_type in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32):
                writer.add_array(name, [int(x) for x in arr_data])
            elif arr_type == gguf.GGUFValueType.FLOAT32:
                writer.add_array(name, [float(x) for x in arr_data])

    # Copy tensors, fixing shapes as needed
    print("Copying tensors...")
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = list(tensor.shape)
        dtype = tensor.tensor_type
        data = tensor.data

        # Fix 5D -> 4D by merging dims 1 and 2 (temporal and in_channels)
        if len(shape) == 5:
            old_shape = shape.copy()
            # [out, temporal, in_ch, h, w] -> [out, temporal*in_ch, h, w]
            shape = [shape[0], shape[1] * shape[2], shape[3], shape[4]]
            print(f"  [{i}] {name}: {old_shape} -> {shape}")

        # Add tensor with potentially fixed shape
        writer.add_tensor(name, data, raw_shape=tuple(shape), raw_dtype=dtype)

    print("Finalizing...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = output_path.stat().st_size
    print(f"Done! Output: {output_path} ({out_size / (1024**3):.1f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Patch GGUF to fix 5D tensors")
    parser.add_argument("input", type=Path, help="Input GGUF file")
    parser.add_argument("-o", "--output", type=Path, help="Output GGUF file (default: input.fixed.gguf)")
    args = parser.parse_args()

    patch_gguf(args.input, args.output)


if __name__ == "__main__":
    main()
