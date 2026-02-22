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
            arr_data = field.parts[-1]
            rope_sections_old = [int(x) for x in arr_data]
            break

    if rope_field_name is None:
        print("No rope.dimension_sections field found, nothing to fix.")
        return

    print(f"  Found: {rope_field_name} = {rope_sections_old}")

    if len(rope_sections_old) >= 4:
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

    # Create new GGUF
    print(f"Writing: {output_path}")
    writer = GGUFWriter(str(output_path), arch=arch)

    # Copy all metadata, overriding rope.dimension_sections
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

                # Override rope.dimension_sections
                if name == rope_field_name:
                    writer.add_array(name, rope_sections_new)
                    print(f"  Patched: {name} = {rope_sections_new}")
                    continue

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

    # Copy all tensors (stream data, don't load into RAM)
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
