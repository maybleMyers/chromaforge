#!/usr/bin/env python3
"""
Patch a GGUF file to fix 5D vision tensors by reshaping to 4D.

Modifies tensor shape metadata without re-quantizing.
[d0, d1, d2, d3, d4] -> [d0, d1*d2, d3, d4] (merge dims 1 and 2)
"""

import argparse
from pathlib import Path

import gguf
from gguf import GGUFReader, GGUFWriter
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

                if t == gguf.GGUFValueType.STRING:
                    if isinstance(val, np.ndarray):
                        s = bytes(val).decode('utf-8').rstrip('\x00')
                    elif isinstance(val, bytes):
                        s = val.decode('utf-8').rstrip('\x00')
                    else:
                        s = str(val)
                    writer.add_string(name, s)
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

    # Build tensor info directly, bypassing shape inference
    print("Adding tensor info...")
    total = len(reader.tensors)

    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = list(tensor.shape)
        dtype = tensor.tensor_type
        data = np.array(tensor.data)  # Copy to ensure contiguous
        n_bytes = len(data.tobytes())

        # Fix 5D -> 4D by merging dims 1 and 2
        if name in tensors_to_fix and len(shape) == 5:
            old_shape = shape.copy()
            # [d0, d1, d2, d3, d4] -> [d0, d1*d2, d3, d4]
            shape = [shape[0], shape[1] * shape[2], shape[3], shape[4]]
            print(f"  [{i+1}/{total}] FIXING {name}: {old_shape} -> {shape}")

        # Directly append to internal structures, bypassing add_tensor_info
        # ti_data entries: (encoded_name, n_dims, shape_array, dtype_value, offset)
        encoded_name = name.encode('utf-8')
        shape_arr = np.array(shape, dtype=np.uint64)
        dtype_val = dtype.value if hasattr(dtype, 'value') else int(dtype)

        # The offset is computed during write, so we use 0 as placeholder
        writer.ti_data.append((encoded_name, len(shape), shape_arr, dtype_val, 0))
        writer.tensors.append(data)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{total} tensors...")

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
