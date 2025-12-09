#!/usr/bin/env python3
import os
import subprocess

os.chdir(os.path.expanduser("~/diffusion/chromaforge/llama-cpp-python"))

# Update submodule to b7142 (commit b61de2b) - same as LM Studio uses
print("Updating llama.cpp submodule to b7142 (b61de2b) - LM Studio version...")
os.chdir("vendor/llama.cpp")
subprocess.run(["git", "fetch", "origin", "--tags"])
subprocess.run(["git", "checkout", "b61de2b"])
os.chdir("../..")

# Patch _ctypes_extensions.py
with open('llama_cpp/_ctypes_extensions.py', 'r') as f:
    content = f.read()

old = """    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function"""

new = """    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                try:
                    func = getattr(lib, name)
                    func.argtypes = argtypes
                    func.restype = restype
                    functools.wraps(f)(func)
                    return func
                except AttributeError:
                    @functools.wraps(f)
                    def stub(*args, **kwargs):
                        raise NotImplementedError(f"Function '{name}' not available")
                    return stub
            else:
                return f

        return decorator

    return ctypes_function"""

if old in content:
    content = content.replace(old, new)
    with open('llama_cpp/_ctypes_extensions.py', 'w') as f:
        f.write(content)
    print('Patched _ctypes_extensions.py!')
else:
    print('Already patched or pattern not found')

# Build
print('Building with CUDA...')
os.environ['CMAKE_ARGS'] = '-DGGML_CUDA=on'
subprocess.run(['pip', 'install', '.', '--force-reinstall', '--no-cache-dir'])
