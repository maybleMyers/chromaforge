#!/usr/bin/env python3
"""
Patch a GGUF file to fix 5D vision tensors by reshaping to 4D.

Modifies tensor shape metadata without re-quantizing.
[d0, d1, d2, d3, d4] -> [d0, d1*d2, d3, d4] (merge dims 1 and 2)
"""

import argparse
from pathlib import Path

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.gguf_writer import TensorInfo
import numpy as np


def patch_gguf(input_path: Path, output_path: Path | None = None):
    """Read GGUF, fix 5D tensors, write new GGUF."""

    if output_path is None:
        output_path = input_path.with_suffix('.fixed.gguf')

    print(f"Reading: {input_path}")
    reader = GGUFReader(str(input_path))

    # Find tensors that need fixing
    tensors_to_fix = set()
    for tensor in reader.tensors:
        if len(tensor.shape) > 4:
            tensors_to_fix.add(tensor.name)
            print(f"  Found 5D tensor: {tensor.name} shape={list(tensor.shape)} dtype={tensor.tensor_type.name}")

    if not tensors_to_fix:
        print("No 5D tensors found, nothing to fix.")
        return

    # Get architecture from metadata
    arch = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            data = field.parts[-1]
            if isinstance(data, np.ndarray):
                arch = bytes(data).decode('utf-8').rstrip('\x00')
            elif isinstance(data, bytes):
                arch = data.decode('utf-8').rstrip('\x00')
            else:
                arch = str(data)
            break

    if arch is None or len(arch) < 2:
        arch = "qwen35moe"
    print(f"  Architecture: {arch}")

    # Create new GGUF with fixed shapes
    print(f"Writing: {output_path}")
    writer = GGUFWriter(str(output_path), arch=arch)

    # Copy all metadata
    print("Copying metadata...")
    for field in reader.fields.values():
        if field.name.startswith('general.') or field.name == 'GGUF.version':
            continue  # Skip auto-added fields

        name = field.name
        parts = field.parts
        types = field.types

        try:
            if len(parts) == 1 and len(types) == 1:
                val = parts[0]
                t = types[0]

                # Handle numpy arrays - extract scalar
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        val = val.flat[0]
                    elif t == gguf.GGUFValueType.STRING:
                        val = bytes(val).decode('utf-8').rstrip('\x00')

                if t == gguf.GGUFValueType.STRING:
                    if isinstance(val, bytes):
                        val = val.decode('utf-8').rstrip('\x00')
                    writer.add_string(name, str(val))
                elif t == gguf.GGUFValueType.UINT32:
                    writer.add_uint32(name, int(val))
                elif t == gguf.GGUFValueType.INT32:
                    writer.add_int32(name, int(val))
                elif t == gguf.GGUFValueType.FLOAT32:
                    writer.add_float32(name, float(val))
                elif t == gguf.GGUFValueType.BOOL:
                    writer.add_bool(name, bool(val))
                elif t == gguf.GGUFValueType.UINT64:
                    writer.add_uint64(name, int(val))
                elif t == gguf.GGUFValueType.INT64:
                    writer.add_int64(name, int(val))
                elif t == gguf.GGUFValueType.FLOAT64:
                    writer.add_float64(name, float(val))
                elif t == gguf.GGUFValueType.UINT8:
                    writer.add_uint8(name, int(val))
                elif t == gguf.GGUFValueType.INT8:
                    writer.add_int8(name, int(val))
                elif t == gguf.GGUFValueType.UINT16:
                    writer.add_uint16(name, int(val))
                elif t == gguf.GGUFValueType.INT16:
                    writer.add_int16(name, int(val))
            elif len(types) >= 2 and types[0] == gguf.GGUFValueType.ARRAY:
                arr_type = types[1]
                arr_data = parts[-1]

                if arr_type == gguf.GGUFValueType.STRING:
                    str_list = []
                    for s in arr_data:
                        if isinstance(s, np.ndarray):
                            str_list.append(bytes(s).decode('utf-8').rstrip('\x00'))
                        elif isinstance(s, bytes):
                            str_list.append(s.decode('utf-8').rstrip('\x00'))
                        else:
                            str_list.append(str(s))
                    writer.add_array(name, str_list)
                elif arr_type in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32):
                    writer.add_array(name, [int(x) for x in arr_data])
                elif arr_type == gguf.GGUFValueType.FLOAT32:
                    writer.add_array(name, [float(x) for x in arr_data])
        except Exception as e:
            print(f"  [WARN] Skipping metadata {name}: {e}")

    # Process tensors - add directly to writer.tensors[0] dict using TensorInfo
    print("Processing tensors...")
    total = len(reader.tensors)
    tensor_dict = writer.tensors[0]  # All tensors go in this dict

    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = tuple(tensor.shape)
        dtype = tensor.tensor_type
        data = tensor.data
        n_bytes = tensor.n_bytes

        # Fix 5D -> 4D by merging dims 1 and 2
        if name in tensors_to_fix and len(shape) == 5:
            old_shape = shape
            # [d0, d1, d2, d3, d4] -> [d0, d1*d2, d3, d4]
            shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
            print(f"  [{i+1}/{total}] FIXING {name}: {list(old_shape)} -> {list(shape)}")

        # Create TensorInfo directly, bypassing shape inference
        # The data stays as raw bytes, we just fix the shape metadata
        tensor_dict[name] = TensorInfo(
            shape=shape,
            dtype=dtype,
            nbytes=n_bytes,
            tensor=np.array(data),  # Ensure it's a proper numpy array
        )

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} tensors...")

    print(f"  Total: {len(tensor_dict)} tensors")

    print("Writing file...")
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
