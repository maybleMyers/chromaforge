# Regional torch.compile, following diffusers' compile_repeated_blocks approach:
# instead of compiling the whole model (slow first-step, fragile guards), compile
# only the repeated transformer blocks. All instances of a block class share one
# compiled code cache, so compile time stays low and the denoise loop runs the
# compiled kernels. Output is numerically equivalent to eager mode.

import torch


# Native (non-diffusers) model classes: attribute names of the ModuleLists that
# hold the repeated transformer blocks. Diffusers classes are handled through
# their own `_repeated_blocks` convention instead.
REPEATED_BLOCK_ATTRS = {
    'IntegratedChromaTransformer2DModel': ('double_blocks', 'single_blocks'),
    'IntegratedChroma2Transformer2DModel': ('double_blocks', 'single_blocks'),
    'IntegratedFluxTransformer2DModel': ('double_blocks', 'single_blocks'),
    'SingleStreamDiT': ('blocks',),
}


def find_repeated_blocks(model):
    for module in model.modules():
        attrs = REPEATED_BLOCK_ATTRS.get(type(module).__name__)
        if attrs is not None:
            blocks = []
            for attr in attrs:
                candidate = getattr(module, attr, None)
                if isinstance(candidate, torch.nn.ModuleList):
                    blocks.extend(candidate)
            return blocks

        repeated_class_names = getattr(module, '_repeated_blocks', None)
        if repeated_class_names:
            return [m for m in module.modules() if type(m).__name__ in repeated_class_names]

    return []


def apply_torch_compile(model):
    from modules.shared import opts

    if not getattr(opts, 'torch_compile', False):
        return model

    compile_mode = getattr(opts, 'torch_compile_mode', 'default')

    blocks = find_repeated_blocks(model)
    if not blocks:
        print(f'Torch compile: no repeated transformer blocks known for {type(model).__name__}, model left in eager mode.')
        return model

    # Device moves (offload round-trips), dtype changes and LoRA load/unload each
    # add a guard variant per block; the default cache limit of 8 is too tight.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 4096)

    # dynamic=True keeps one graph across resolution / prompt-length changes.
    # fullgraph stays off so Forge's runtime weight handling (manual cast,
    # online LoRA merge, quantized ops) can graph-break instead of erroring.
    for block in blocks:
        block.compile(mode=compile_mode, dynamic=True)

    print(f'Torch compile: compiled {len(blocks)} repeated transformer blocks of {type(model).__name__} (mode={compile_mode}).')
    return model
